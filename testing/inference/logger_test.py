import contextlib
import io
import os
import shutil
import unittest

from mxfusion.inference.logger import Logger


class LoggerTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.inference.Logger methods.
    """
    def setUp(self):
        self.log_dir = 'log_dir'
        self.log_name = 'log_name'
        self.tag = 'tag'

    def tearDown(self):
        if self.log_dir in os.listdir():
            shutil.rmtree(self.log_dir)

    def test_validate_log_args(self):
        Logger._validate_log_args(self.log_dir, self.log_name)
        Logger._validate_log_args(self.log_dir, None)
        Logger._validate_log_args(None, None)

        with self.assertRaises(ValueError):
            Logger._validate_log_args(None, self.log_name)

    def test_get_board_path__no_log_name(self):
        path = Logger._get_board_path(self.log_dir, None)

        assert path.startswith(self.log_dir)
        assert len(path) == 25, len(path)

    def test_get_board_path__with_log_name(self):
        path = Logger._get_board_path(self.log_dir, self.log_name)

        assert path == self.log_dir + '/' + self.log_name, path

        assert os.path.isdir(path)

    def test_get_board_path__duplicates(self):
        path1 = Logger._get_board_path(self.log_dir, self.log_name)
        path2 = Logger._get_board_path(self.log_dir, self.log_name)

        assert path1 != path2

    def test_verbose(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f), Logger() as ll:
            ll.log(self.tag, 404, 2)
        s = f.getvalue()
        assert s == '\rIteration 2 tag: 404.000\t\t\t\t\n'

    def test_verbose_multi(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f), Logger() as ll:
            ll.log(self.tag, 404, 2)
            ll.log(self.tag, 808, 3)
        s = f.getvalue()
        assert s == '\rIteration 2 tag: 404.000\t\t\t\t\rIteration 3 tag: 808.000\t\t\t\t\n'

    def test_verbose_multi_newline(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f), Logger() as ll:
            ll.log(self.tag, 404, 2, newline=True)
            ll.log(self.tag, 808, 3)
        s = f.getvalue()
        assert s == '\rIteration 2 tag: 404.000\t\t\t\t\n\rIteration 3 tag: 808.000\t\t\t\t\n'

    def test_board_log(self):
        with Logger(verbose=False, log_dir=self.log_dir, log_name=self.log_name) as ll:
            ll.log(self.tag, 404, 2)
        path = os.path.join(self.log_dir, self.log_name)
        assert os.path.isdir(path)
        assert len(os.listdir(path)) == 1

    def test_flush(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f), Logger() as ll:
            ll._on_new_line = False
            print('\rabc', end='')
            ll.flush()
            print('\r123', end='')
        s = f.getvalue()
        assert s == '\rabc\n\r123'
