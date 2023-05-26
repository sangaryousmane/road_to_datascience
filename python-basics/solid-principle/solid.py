"""
Implement the Solid principle in OOD
1. Single Responsibility Principle (SRP)
"""

from zipfile import ZipFile
from pathlib import Path

class FileManager:

    def __int__(self, filename):
        self.path = Path(filename);


    def read(self, encoding='utf-8'):
        self.path.read_text(encoding);

    def write(self, data, encoding='utf-8'):
        self.path.write_text(dat, encoding);


class ZipFileManager:

    def __init__(self, filename):
        self.path = Path(filename);

    def compress(self):
        with ZipFile(self.path.with_suffix('.zip'), mode='w') as archive:
            archive.write(self.path);

    def decompress(self):
        with zipfile(self.path.with_suffix('.zip'), mode='r') as archive:
            archive.extractall()


