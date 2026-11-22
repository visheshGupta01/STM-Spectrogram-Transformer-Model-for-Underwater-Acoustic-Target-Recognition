import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Inspect the course-provided ShipsEar Excel metadata.")
    p.add_argument("excel_file")
    a = p.parse_args()
    book = pd.ExcelFile(a.excel_file)
    print("Sheets:", book.sheet_names)
    for sheet in book.sheet_names:
        frame = pd.read_excel(a.excel_file, sheet_name=sheet)
        print(f"\n--- {sheet}: {frame.shape} ---")
        print("Columns:", list(frame.columns))
        print(frame.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
