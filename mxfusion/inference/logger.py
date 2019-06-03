# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


import os
import time


class Logger:
    """
    The class for logging the results of optimization.

    :param boolean verbose: whether to print per-iteration messages
    :param str log_dir: the directory in which to place the tensorboard logs directory. If this is not set then no
                        tensorboard logs will be written
    :param str log_name: the directory in which to place tensoreboard logs. If no name is assigned, a timestamp name
                         will be used.
    """

    def __init__(self, verbose=True, log_dir=None, log_name=None):
        self.verbose = verbose

        self._validate_log_args(log_dir, log_name)
        self.log_dir = log_dir
        self.log_name = log_name

        self.summary_writer = None
        self._on_new_line = True

    @staticmethod
    def _validate_log_args(log_dir, log_name):
        if log_name and log_dir is None:
            raise ValueError("No log directory provided for MXBoard. Log name given: {}".format(log_name))

    def open(self):
        """
        Open logger.
        """
        self.summary_writer = self.log_dir and self._get_board(self.log_dir, self.log_name)
        self._on_new_line = True

    def close(self):
        """
        Close logger.
        """
        if not self._on_new_line:
            print()
        self.summary_writer and self.summary_writer.close()

    @staticmethod
    def _get_board_path(log_dir, log_name):
        log_name = time.strftime("%Y%m%d-%H:%M:%S", time.localtime()) if log_name is None else log_name

        path = os.path.join(log_dir, log_name)

        i = 1
        _temp_path = path
        while os.path.isdir(path):
            path = '{}_{}'.format(_temp_path, i)
            i += 1

        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _get_board(log_dir, log_name):
        # Only import mxboard if absolutely necessary
        from mxboard import SummaryWriter

        return SummaryWriter(Logger._get_board_path(log_dir, log_name))

    def log(self, tag, value, step, iterate_name='Iteration', precision=3, newline=False):
        """
        Log value.

        :param str tag: name for the logged value
        :param value: value to log
        :type value: float, tuple, list, or dict
        :param int step: step value to log
        :param str iterate_name: name of the iterate
        :param int precision: number of decimal places to show
        :param boolean newline: whether to terminate log with newline
        """
        if self.verbose:
            self._on_new_line = newline
            print('\r{} {} {}: {:.{precision}f}\t\t\t\t'.format(
                iterate_name, step, tag, value, precision=precision), end='\n' if newline else '')

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(tag=tag, value=value, global_step=step)

    def flush(self):
        """
        Flushes board writer and adds new line if not already on a new line.
        """
        if not self._on_new_line and self.verbose:
            print()
        self._on_new_line = True
        if self.summary_writer:
            self.summary_writer.flush()
