

if __name__ == "__main__":
    text = "--fs 22050 --audio-format flac --segment=dump/raw/org/tr_no_dev/logs/segments.2 data/tr_no_dev/wav.scp dump/raw/org/tr_no_dev/data/format.2"
    arr = text.split(" ")

    print("[" + ", ".join(["\"{}\"".format(tmp) for tmp in arr]) + "]")