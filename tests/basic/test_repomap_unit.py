import difflib
import os
import re
import time
import unittest
from pathlib import Path

import git

from aider.dump import dump  # noqa: F401
from aider.io import InputOutput
from aider.models import Model
from aider.repomap_unit import RepoMapUnit
from aider.utils import GitTemporaryDirectory, IgnorantTemporaryDirectory


class TestRepoMapUnit(unittest.TestCase):
    def setUp(self):
        self.GPT35 = Model("gpt-3.5-turbo")

    def test_get_repo_map_with_identifiers(self):
        io = InputOutput()
        current_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_path, 'data')
        dump(data_path)
        repo_map = RepoMapUnit(main_model=self.GPT35, root=data_path, io=io)

        chat_files = [data_path+'/Person.java']
        other_files = [data_path+'/Address.java',
                       data_path+'/Company.java',
                       data_path+'/NoRef.java']
        result = repo_map.get_repo_map(chat_files, other_files)
        dump(result)

if __name__ == "__main__":
    unittest.main()
