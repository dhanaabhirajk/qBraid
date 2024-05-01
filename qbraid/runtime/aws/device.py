# Copyright (C) 2023 qBraid
#
# This file is part of the qBraid-SDK
#
# The qBraid-SDK is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the qBraid-SDK, as per Section 15 of the GPL v3.

"""
Module defining BraketDeviceWrapper Class

"""
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Union

import braket.circuits
from braket.aws import AwsDevice
from braket.device_schema import ExecutionDay

from qbraid.programs.libs.braket import BraketCircuit
from qbraid.runtime.device import QuantumDevice
from qbraid.runtime.enums import DeviceStatus, DeviceType
from qbraid.runtime.profile import RuntimeProfile
from qbraid.transpiler import CircuitConversionError, ConversionPathNotFoundError, transpile

from .job import BraketQuantumTask

if TYPE_CHECKING:
    import braket.aws

    import qbraid.runtime.aws
    import qbraid.transpiler


def _future_utc_datetime(hours: int, minutes: int, seconds: int) -> str:
    """Return a UTC datetime that is hours, minutes, and seconds from now
    as an ISO string without fractional seconds."""
    current_utc = datetime.utcnow()
    future_time = current_utc + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return future_time.strftime("%Y-%m-%dT%H:%M:%SZ")


class BraketDevice(QuantumDevice):
    """Wrapper class for Amazon Braket ``Device`` objects."""

    def __init__(
        self,
        arn: Optional[str] = None,
        profile: "Optional[RuntimeProfile]" = None,
        provider: "Optional[qbraid.runtime.aws.BraketProvider]" = None,
        device: "Optional[braket.aws.AwsDevice]" = None,
    ):
        """Create a BraketDevice."""
        if not (arn or device):
            raise ValueError("Must specify either arn or device")
        if arn and device:
            raise ValueError("Can only specify one of arn and device")
        if not (profile or provider):
            raise ValueError("Must specify either profile or provider")
        super().__init__(device_id=arn or device.arn, provider=provider)

    def _get_device(
        self, arn: str, provider: "Optional[qbraid.runtime.aws.BraketProvider]" = None
    ) -> "braket.aws.AwsDevice":
        """Return the Braket device object."""
        if provider:
            region_name = provider._get_region_name(arn)
            aws_session = provider._get_aws_session(region_name=region_name)
        else:
            aws_session = None
        return AwsDevice(arn=arn, aws_session=aws_session)

    def _default_profile(self) -> "qbraid.runtime.RuntimeProfile":
        """Return the default runtime profile."""
        return RuntimeProfile(
            device_type=self._device_type,
            device_num_qubits=self._num_qubits,
            program_spec=self._program_spec,
        )

    def status(self) -> "qbraid.runtime.DeviceStatus":
        """Return the status of this Device."""
        if self._device.status == "ONLINE":
            if self._device.is_available:
                return DeviceStatus.ONLINE
            return DeviceStatus.UNAVAILABLE

        if self._device.status == "RETIRED":
            return DeviceStatus.RETIRED

        return DeviceStatus.OFFLINE

    def availability_window(self) -> tuple[bool, str, str]:
        """Provides device availability status. Indicates current availability,
        time remaining (hours, minutes, seconds) until next availability or
        unavailability, and future UTC datetime of next change in availability status.

        Returns:
            tuple[bool, str, str]: Current device availability, hr/min/sec until availability
                                   switch, future UTC datetime of availability switch
        """

        is_available_result = False
        available_time = None

        device = self._device

        if device.status != "ONLINE":
            return is_available_result, "", ""

        day = 0

        current_datetime_utc = datetime.utcnow()
        for execution_window in device.properties.service.executionWindows:
            weekday = current_datetime_utc.weekday()
            current_time_utc = current_datetime_utc.time().replace(microsecond=0)

            if (
                execution_window.windowEndHour < execution_window.windowStartHour
                and current_time_utc < execution_window.windowEndHour
            ):
                weekday = (weekday - 1) % 7

            matched_day = execution_window.executionDay == ExecutionDay.EVERYDAY
            matched_day = matched_day or (
                execution_window.executionDay == ExecutionDay.WEEKDAYS and weekday < 5
            )
            matched_day = matched_day or (
                execution_window.executionDay == ExecutionDay.WEEKENDS and weekday > 4
            )
            ordered_days = (
                ExecutionDay.MONDAY,
                ExecutionDay.TUESDAY,
                ExecutionDay.WEDNESDAY,
                ExecutionDay.THURSDAY,
                ExecutionDay.FRIDAY,
                ExecutionDay.SATURDAY,
                ExecutionDay.SUNDAY,
            )
            matched_day = matched_day or (
                execution_window.executionDay in ordered_days
                and ordered_days.index(execution_window.executionDay) == weekday
            )

            matched_time = (
                execution_window.windowStartHour < execution_window.windowEndHour
                and execution_window.windowStartHour
                <= current_time_utc
                <= execution_window.windowEndHour
            ) or (
                execution_window.windowEndHour < execution_window.windowStartHour
                and (
                    current_time_utc >= execution_window.windowStartHour
                    or current_time_utc <= execution_window.windowEndHour
                )
            )

            if execution_window.executionDay in ordered_days:
                day = (ordered_days.index(execution_window.executionDay) - weekday) % 6

            start_time = execution_window.windowStartHour
            end_time = current_time_utc

            if day == 0:
                day_factor = day
            elif start_time.hour < end_time.hour:
                day_factor = day - 1
            elif start_time.hour == end_time.hour:
                if start_time.minute < end_time.minute:
                    day_factor = day - 1
                elif start_time.minute == end_time.minute:
                    if start_time.second <= end_time.second:
                        day_factor = day - 1
            else:
                day_factor = day

            td = timedelta(
                hours=start_time.hour,
                minutes=start_time.minute,
                seconds=start_time.second,
            ) - timedelta(hours=end_time.hour, minutes=end_time.minute, seconds=end_time.second)
            days_seconds = 86400 * day_factor
            available_time_secs = td.seconds + days_seconds

            if available_time is None or available_time_secs < available_time:
                available_time = available_time_secs

            is_available_result = is_available_result or (matched_day and matched_time)

        if available_time is None:
            return is_available_result, "", ""

        hours = available_time // 3600
        minutes = (available_time // 60) % 60
        seconds = available_time - hours * 3600 - minutes * 60
        available_time_hms = f"{hours:02}:{minutes:02}:{seconds:02}"
        utc_datetime_str = _future_utc_datetime(hours, minutes, seconds)
        return is_available_result, available_time_hms, utc_datetime_str

    def queue_depth(self) -> int:
        """Return the number of jobs in the queue for the device."""
        queue_depth_info = self._device.queue_depth()
        total_queued = 0
        for queue_type in ["Normal", "Priority"]:
            num_queued = queue_depth_info.quantum_tasks[queue_type]
            if isinstance(num_queued, str) and num_queued.startswith(">"):
                num_queued = num_queued[1:]
            total_queued += int(num_queued)
        return total_queued

    def transform(self, run_input: "braket.circuits.Circuit") -> "braket.circuits.Circuit":
        """Transpile a circuit for the device."""
        if self.device_type == DeviceType.SIMULATOR:
            program = BraketCircuit(run_input)
            program.remove_idle_qubits()
            run_input = program.program

        if self._provider_name == "ionq":
            graph = self.profile.get("conversion_graph")
            if graph is not None and graph.has_edge("pytket", "braket"):
                # pylint: disable=import-outside-toplevel
                from qbraid.transforms.pytket.ionq import pytket_ionq_transform

                try:
                    tk_circuit = transpile(run_input, "pytket", max_path_depth=1)
                    tk_transformed = pytket_ionq_transform(tk_circuit)
                    run_input = transpile(tk_transformed, "braket", max_path_depth=1)
                except (ConversionPathNotFoundError, CircuitConversionError):
                    pass

        return run_input

    def submit(
        self,
        run_input: Union[braket.circuits.Circuit, list[braket.circuits.Circuit]],
        *args,
        **kwargs,
    ) -> Union[BraketQuantumTask, list[BraketQuantumTask]]:
        """Run a quantum task specification on this quantum device. Task must represent a
        quantum circuit, annealing problems not supported.

        Args:
            run_input: Specification of a task to run on device.

        Keyword Args:
            shots (int): The number of times to run the task on the device.

        Returns:
            The job like object for the run.

        """
        is_single_input = not isinstance(run_input, list)
        run_input = [run_input] if is_single_input else run_input
        aws_quantum_task_batch = self._device.run_batch(run_input, *args, **kwargs)
        tasks = [
            BraketQuantumTask(task.metadata()["quantumTaskArn"], device=self._device)
            for task in aws_quantum_task_batch.tasks
        ]
        if is_single_input:
            return tasks[0]
        return tasks
