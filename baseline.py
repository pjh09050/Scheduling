import sys
from problem_input.problem import *
from simulation.sim_util import *
import json
from datetime import datetime, timedelta


class Simulator(SimUtil):

    def __init__(self, production_instance: "ProblemInstance", exp_json, sim_json):
        super().__init__()
        self.exp_parameters = json.load(open(exp_json))
        self.sim_parameters = json.load(open(sim_json))
        self.instance = cp(production_instance)
        self.index = self.instance.index

    def run(self):
        start_time = datetime.now()
        assert isinstance(self.instance, ProblemInstance)
        self.initialize()

        while self.process_event():
            pass

        # print('Test Finish', self.event_id)
        print(f"taken_time={datetime.now() - start_time}")
        print('Utilization', self.KPIs.calculate_utilization())
        print('AVG TAT', self.KPIs.calculate_TAT())

        self.write_history()

    def initialize(self):
        self.set_sim_clock(time.time())
        self.T = 0
        self.instance.event_list.sort(key=lambda e: (e.start_time + e.period_time, e.lot_id))

        # Process event list from DB
        for event in self.instance.event_list:
            equipment_id = event.equipment_id
            current_equipment = self.instance.equipment_map[equipment_id]
            assert isinstance(current_equipment, Equipment)
            if event.event_type == self.TrackInFinish:
                current_equipment.last_work_finish_time = event.start_time + event.period_time
                self.instance.lot_map[event.lot_id].now_event = event

        for current_lot_id, current_lot in self.instance.lot_map.items():
            self.total_num_of_lot += 1
            assert isinstance(current_lot, Lot)
            device_id = current_lot.device_id
            current_device = self.instance.device_map[device_id]
            assert isinstance(current_device, Device)
            flow_number = current_lot.flow_number
            current_operation_id = current_device.operation_sequence[flow_number]
            current_location = current_lot.location
            if current_location == self.PREV_STOCKER:
                self.lot_list_in_prev_stock.append(current_lot)
            elif current_location == self.STOCKER:
                self.lot_list_in_stock.append(current_lot)

            # WIP Level Update
            self.counters.WIP_level_per_device[device_id][current_operation_id] += 1
            self.WIP_level_per_T_device[device_id][current_operation_id][self.T] += 1

        # Dispatching Decision
        self.equipment_list_with_empty_buffer = list(self.instance.equipment_map.values())
        available_lot_list, available_eqp_list = self.update_available_lot_equipment_list()
        while len(available_eqp_list) > 0:
            if self.dispatching_lot_to_equipment(self.exp_parameters['dispatchingPolicy'], available_eqp_list, available_lot_list) is False:
                break

        self.instance.event_list.sort(key=lambda e: (e.start_time + e.period_time, e.lot_id))

    def process_event(self):
        if len(self.instance.event_list) == 0:
            self.KPIs.makespan = self.T
            return False
        # print(self.T, len(self.lot_list_in_prev_stock))

        event = self.instance.event_list.pop(0)
        event_type = event.event_type
        assert isinstance(event, Event)
        self.T = event.start_time + event.period_time
        # print(self.T, len(self.instance.event_list))
        if event_type == self.FactoryInFinish:
            self.factory_in_finish(event)
        elif event_type == self.MoveToStockerFinish:
            self.move_to_stocker_finish(event)
        elif event_type == self.TrackInFinish:
            self.track_in_finish(event)
        elif event_type == self.MoveToEquipmentFinish:
            self.move_to_equipment_finish(event)
        elif event_type == self.SetupChangeFinish:
            self.setup_change_finish(event)
        elif event_type == 'DataCollectingFinish':
            pass
        self.instance.event_list.sort(key=lambda e: (e.start_time + e.period_time, e.lot_id))
        return True

    def factory_in_finish( self, event: "Event"):
        current_lot = self.instance.lot_map[event.lot_id]
        assert isinstance(current_lot, Lot)
        current_operation = current_lot.current_operation_id
        current_lot.record_history(self.FACTORY_IN, self.T, self.WAIT, self.PREV_STOCKER, '', current_operation)
        # waiting_event = Event(self.give_event_number(), current_operation, current_lot.id, '', self.Waiting, self.T, sys.maxsize)
        # self.instance.event_list.append(waiting_event)

    def track_in_finish(self, event: "Event"):
        self.KPIs.makespan = self.T
        current_lot = self.instance.lot_map[event.lot_id]
        current_device = self.instance.device_map[current_lot.device_id]
        current_equipment = self.instance.equipment_map[event.equipment_id]
        assert isinstance(current_lot, Lot)
        assert isinstance(current_device, Device)
        assert isinstance(current_equipment, Equipment)
        current_flow_num = current_lot.flow_number
        current_operation_id = event.operation_id
        current_equipment.record_history(self.END_PROCESS, self.T, self.IDLE, '', '', '', '')
        self.counters.cumulative_output_per_operation[current_device.device_id][current_operation_id] += current_lot.quantity
        # Depends on whether the current operation is ths last one or not
        if current_flow_num == len(current_device.operation_sequence) - 1:
            current_lot.record_history(self.TRACK_OUT, self.T, self.WAIT, 'END', '', 'SHIP')
            current_lot.record_history(self.FACTORY_OUT, self.T, self.WAIT, 'END', '', '')
            self.KPIs.TAT_per_lot[current_lot.id] = self.T - current_lot.factory_in_time
            if self.T < current_lot.due_date:
                self.KPIs.completed_product_quantity[current_device.device_id] += current_lot.quantity
                self.counters.ship_count += 1
                self.counters.ship_count_per_T[self.T] = self.counters.ship_count
        else:
            current_lot.flow_number += 1
            next_operation_id = current_device.operation_sequence[current_lot.flow_number]
            current_lot.record_history(self.TRACK_OUT, self.T, self.WAIT, self.WAY_TO_STOCKER, '', next_operation_id)
            current_lot.record_history(self.MOVE_START, self.T, self.MOVE, self.WAY_TO_STOCKER, '', next_operation_id)
            current_lot.current_operation_arrival_time = self.T
            current_lot.current_operation_start_time = -1
            move_event = Event(self.add_event_number(), next_operation_id, current_lot.id, '', self.MoveToStockerFinish, self.T, self.sim_parameters['moving_time'])
            self.instance.event_list.append(move_event)
        # KPI update
        if event.start_time < self.termination_time < self.T:
            temp_time = max(self.termination_time - event.start_time, 0)
            self.KPIs.u_per_equipment[event.equipment_id] += temp_time
        elif self.T < self.termination_time:
            self.KPIs.u_per_equipment[event.equipment_id] += event.period_time

        #FIXME!!!
        # 물리적으로 buffer에 lot이 들어갔을 때와 할당이 되었을 때 buffer에 lot을 할당하는 것을 명확히 구분해야함
        # 일단은 물리적으로 buffer에 lot이 들어갔을 때 buffer에 lot을 append하고, self.equipment_empty_buffer는 따로 운영하는 것이 좋을듯
        # 추후에 이동시간이 장비 위의 작업시간보다 길어지는 상황 고려해야 함
        if len(current_equipment.buffer) > 0:
            lot_in_buffer_id = current_equipment.buffer[0]
            lot_in_buffer = self.instance.lot_map[lot_in_buffer_id]
            assert isinstance(lot_in_buffer, Lot)
            setup_time = self.get_setup_time(current_equipment, lot_in_buffer)
            if setup_time == 0:
                self.update_lot_equipment_status_after_starting_track_in(lot_in_buffer, current_equipment)
                self.flush_equipment_buffer(current_equipment)
                available_lot_list, available_eqp_list = self.update_available_lot_equipment_list()
                while len(available_eqp_list) > 0:
                    if self.dispatching_lot_to_equipment(self.exp_parameters['dispatchingPolicy'], available_eqp_list, available_lot_list) is False:
                        break
            else:
                lot_in_buffer.record_history(self.START_SETUP, self.T, self.PROCESS, event.equipment_id, event.equipment_id)
                current_equipment.record_history(self.START_SETUP,  self.T, self.SETUP, "", "", "", "")
                lot_id = lot_in_buffer.id
                setup_event = Event(self.add_event_number(), current_operation_id, lot_id, current_equipment.id, self.SetupChangeFinish, self.T, setup_time)
                self.instance.event_list.append(setup_event)
                self.flush_equipment_buffer(current_equipment)


        # Gantt chart..
        # self.ganttChart.scheduleForResource[event.resourceId].append(
        #     Gantt(aa, event.start_time, (event.start_time + event.period_time), product_id, event.lotId,
        #           nowLot.lotQuantity, nowLot.currentOperationId, nowLot.flowNumber))

    # Make a dispatching decision for assigning a lot to an equipment
    def move_to_stocker_finish(self, event: "Event"):
        current_lot = self.instance.lot_map[event.lot_id]
        assert isinstance(current_lot, Lot)
        current_lot.record_history(self.MOVE_END, self.T, self.WAIT, self.STOCKER, '')
        self.lot_list_in_stock.append(current_lot)
        available_lot_list, available_eqp_list = self.update_available_lot_equipment_list()
        while len(available_eqp_list) > 0:
            if self.dispatching_lot_to_equipment(self.exp_parameters['dispatchingPolicy'], available_eqp_list, available_lot_list) is False:
                break

    def move_to_equipment_finish(self, event: "Event"):
        lot_id = event.lot_id
        current_lot = self.instance.lot_map[lot_id]
        assert isinstance(current_lot, Lot)
        device_id = current_lot.device_id
        current_operation_id = current_lot.current_operation_id
        equipment_id = event.equipment_id
        current_lot.record_history(self.MOVE_END, self.T, self.WAIT, equipment_id + '_BUFFER', equipment_id)

        current_equipment = self.instance.equipment_map[event.equipment_id]
        assert isinstance(current_equipment, Equipment)
        current_equipment.buffer.append(lot_id)
        setup_time = self.get_setup_time(current_equipment, current_lot)

        if current_equipment.status == self.IDLE and setup_time == 0:
            self.update_lot_equipment_status_after_starting_track_in(current_lot, current_equipment)
            self.flush_equipment_buffer(current_equipment)
            available_lot_list, available_eqp_list = self.update_available_lot_equipment_list()
            while len(available_eqp_list) > 0:
                if self.dispatching_lot_to_equipment(self.exp_parameters['dispatchingPolicy'], available_eqp_list, available_lot_list) is False:
                    break
        else:
            total_time = self.get_processing_time(current_equipment, current_lot)
            current_equipment.processing_time_in_buffer = total_time
            if current_equipment.status == self.IDLE and setup_time > 0:
                current_lot.record_history(self.START_SETUP, self.T, self.PROCESS, event.equipment_id, event.equipment_id)
                current_equipment.record_history(self.START_SETUP, self.T, self.SETUP, "", "", "", "")
                setup_event = Event(self.add_event_number(), current_operation_id, lot_id, equipment_id, self.SetupChangeFinish, self.T, setup_time)
                self.instance.event_list.append(setup_event)
                self.flush_equipment_buffer(current_equipment)

    def setup_change_finish(self, event: "Event"):
        equipment_id = event.equipment_id
        current_equipment = self.instance.equipment_map[equipment_id]
        current_lot = self.instance.lot_map[event.lot_id]
        assert isinstance(current_equipment, Equipment)
        assert isinstance(current_lot, Lot)
        self.set_setup_status(current_lot, current_equipment)
        current_lot.record_history(self.END_SETUP, self.T, self.WAIT, event.equipment_id, event.equipment_id)
        current_equipment.record_history(self.END_SETUP, self.T, self.IDLE, "", "", "", "")
        current_equipment.last_work_finish_time = self.T

        # Current equipment should has a lot in its buffer
        self.update_lot_equipment_status_after_starting_track_in(current_lot, current_equipment)
        available_lot_list, available_eqp_list = self.update_available_lot_equipment_list()
        while len(available_eqp_list) > 0:
            if self.dispatching_lot_to_equipment(self.exp_parameters['dispatchingPolicy'], available_eqp_list, available_lot_list) is False:
                break

    def dispatching_lot_to_equipment(self, dispatchingPolicy, equipment_list, lot_list):
        assigned_lot_id, equipment_id = self.dispatching_rule(lot_list, dispatchingPolicy, equipment_list)
        if assigned_lot_id != '' and equipment_id != '':
            assigned_lot = self.instance.lot_map[assigned_lot_id]
            current_equipment = self.instance.equipment_map[equipment_id]
            assert isinstance(assigned_lot, Lot)
            operation_id = assigned_lot.current_operation_id
            moving_event = Event(self.add_event_number(), operation_id, assigned_lot_id, equipment_id, self.MoveToEquipmentFinish, self.T, self.sim_parameters['moving_time'])
            self.instance.event_list.append(moving_event)
            # 물리적으로 아직 할당 된 것은 아니므로 주석처리
            # current_equipment.buffer.append(assigned_lot_id)

            self.remove_lot_equipment_from_available(assigned_lot, current_equipment, lot_list, equipment_list)
            self.remove_lot_equipment_from_candidates(assigned_lot, current_equipment)

            # 첫 번째 할당인지 아닌지 여부
            if assigned_lot.flow_number > 0:
                latest_lot_history = assigned_lot.history_list[-1]
                assert isinstance(latest_lot_history, LotHistory)
                self.KPIs.w_per_lot[assigned_lot_id] += self.T - latest_lot_history.event_time
            else:
                self.KPIs.w_per_lot[assigned_lot_id] += self.T - assigned_lot.factory_in_time

            to_location = current_equipment.id + '_' + 'BUFFER'
            assigned_lot.record_history(self.MOVE_START, self.T, self.MOVE, to_location, current_equipment.id)
            # Update WIP Status (물리적으로 아직 할당 된 것이 아니므로 수행하지 않음)
            # Learner 혹은 의사결정을 위한 상태를 표현할 때는 물리적인 것과는 별개의 update를 수행해야 할 수 있음
            return True
        else:
            return False

    def remove_lot_equipment_from_candidates(self, lot: "Lot", equipment: "Equipment"):
        if lot in self.lot_list_in_prev_stock:
            self.lot_list_in_prev_stock.remove(lot)
        elif lot in self.lot_list_in_stock:
            self.lot_list_in_stock.remove(lot)
        else:
            print("ERROR in remove lot from candidates")
        self.equipment_list_with_empty_buffer.remove(equipment)

    def remove_lot_equipment_from_available(self, lot: "Lot", equipment: "Equipment", lot_list, equip_list):
        lot_list.remove(lot)
        equip_list.remove(equipment)

    def update_lot_equipment_status_after_starting_track_in(self, current_lot: "Lot", current_equipment:"Equipment"):
        current_lot.current_operation_start_time = self.T
        lot_id = current_lot.id
        equipment_id = current_equipment.id
        prior_flow_index = current_lot.flow_number - 1
        if prior_flow_index < 0:
            prior_operation_id = 'IN'
        else:
            prior_operation_id = self.instance.device_map[current_lot.device_id].operation_sequence[current_lot.flow_number - 1]
        current_operation_id = current_lot.current_operation_id
        current_lot.record_history(self.TRACK_IN, self.T, self.PROCESS, equipment_id, equipment_id)
        current_equipment.record_history(self.START_PROCESS, self.T, self.RUN, current_lot.device_id, lot_id, current_operation_id, prior_operation_id)
        total_time = self.get_processing_time(current_equipment, current_lot)
        current_equipment.last_work_finish_time = self.T + total_time
        track_in_event = Event(self.add_event_number(), current_operation_id, lot_id, equipment_id, self.TrackInFinish, self.T, total_time)
        self.instance.event_list.append(track_in_event)

    def update_available_lot_equipment_list(self):
        available_equipment_list = []
        available_lot_list = []
        for eqp in self.equipment_list_with_empty_buffer:
            eqp_exist_flag = False
            assert isinstance(eqp, Equipment)
            eqp_group_id = eqp.equipment_group_id
            eqp_group = self.instance.equipment_group_map[eqp_group_id]
            assert isinstance(eqp_group, EquipmentGroup)
            eqp_arrange_map = eqp_group.equipment_arrangement_map
            for lot in self.lot_list_in_stock + self.lot_list_in_prev_stock:
                assert isinstance(lot, Lot)
                device_id = lot.device_id
                if device_id in eqp_arrange_map:
                    if lot.current_operation_id in eqp_arrange_map[device_id]:
                        if eqp_exist_flag is False:
                            available_equipment_list.append(eqp)
                            eqp_exist_flag = True
                        if lot not in available_lot_list:
                            available_lot_list.append(lot)

        return available_lot_list, available_equipment_list

    def flush_equipment_buffer(self, current_equipment):
        current_equipment.processing_time_in_buffer = 0
        self.equipment_list_with_empty_buffer.append(current_equipment)









