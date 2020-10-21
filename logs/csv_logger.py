import csv
import os

class CSVLogger():
    def __init__(self, log_path: str, headers: list):
        self.log_path = log_path
        self.headers = headers

    def write_header(self):
        f = open(self.log_path, "w")
        with f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def write_row(self, log_entry_dict):
        f = open(self.log_path, "a")
        with f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(log_entry_dict)

    