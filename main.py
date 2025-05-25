from versionk.rule_extractor import entry_finder

if __name__ == "__main__":
    entry_finder(
        csv_file = 'data/GBPNZ_H12.csv',
        expose_days = 4,
        threshold = 65,
        short = True,
        year_start = 2016,
        year_end = 2019
    )