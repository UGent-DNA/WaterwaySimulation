import os
from typing import List

from configurations import OUTPUT_PATH


class TableWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def write_table(self, header: List[str], rows: List[List[str]], table_type: str):
        extensions = {"latex": ".tex", "markdown": ".md"}
        base_path = os.path.join(OUTPUT_PATH, "tables", table_type)
        os.makedirs(base_path, exist_ok=True)
        with open(os.path.join(base_path, self.filename + extensions[table_type]), "w") as f:
            self.file = f
            if table_type == "markdown":
                self._print_header(header)
                for row in rows:
                    self._print_table_line(row)
            elif table_type == "latex":
                self._print_header_latex(header)
                for row in rows:
                    self._print_line_latex(row)
                self._print_end_latex()
            else:
                raise NotImplementedError(f"Table type {table_type} was not implemented")

    def _print_header(self, names: List[str]):
        self._print_table_line(names)
        self._print_table_line(['---'] * len(names))

    def _print_table_line(self, line):
        self.file.write(f"| {' | '.join(line)} |\n")

    def _print_header_latex(self, line: List[str]):
        self.file.write("\\begin{tabular}{@{}" + "l" + "r" * (len(line) - 1) + "@{}}\n")
        self.file.write("\t\\toprule\n")
        self._print_line_latex(line)
        self.file.write("\t\\midrule\n")

    def _print_line_latex(self, line: List[str]):
        self.file.write("\t" + " & ".join(line) + "\\\\" + "\n")

    def _print_end_latex(self):
        self.file.write("\t\\bottomrule\n")
        self.file.write("\\end{tabular}\n")
