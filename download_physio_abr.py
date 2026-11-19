"""Download raw ABR only from PhysioNet earnDB (Python standard library only).

Run: py download_physio_abr.py
Test one record: py download_physio_abr.py --limit 1
Output is single-channel WFDB .dat/.hea.
ABR samples and calibration are preserved exactly; no averaging or filtering.
Mixed files must travel over the network, but OAE samples are never saved.
Source: https://physionet.org/content/earndb/1.0.0/
Please cite Silva & Epstein (2010) and PhysioNet when using these data.
"""

import argparse
from pathlib import Path, PurePosixPath
import sys
import time
from urllib.request import urlopen

BASE = "https://physionet.org/files/earndb/1.0.0/"


def fetch(path):
    with urlopen(BASE + path, timeout=120) as response:
        return response.read()


def download_record(record, destination):
    lines = fetch(record + ".hea").decode("ascii").splitlines()
    first = lines[0].split()
    count, samples = int(first[1]), int(first[3])
    signals = [line.split() for line in lines[1:count + 1]]
    abr = [i for i, signal in enumerate(signals)
           if signal[2].split("/")[-1] == "nV"]
    if not abr:
        return "OAE only; skipped"
    if len(abr) != 1:
        raise ValueError("Expected exactly one ABR channel")
    index = abr[0]
    fmt = signals[index][1]
    if fmt not in {"16", "24", "32"} or any(
        s[0] != signals[index][0] or s[1] != fmt for s in signals
    ):
        raise ValueError("Unsupported signal layout; refusing to alter samples")
    width = int(fmt) // 8
    folder = destination / PurePosixPath(record).parent
    folder.mkdir(parents=True, exist_ok=True)
    name = PurePosixPath(record).name
    dat = folder / (name + ".dat")
    hea = folder / (name + ".hea")
    if (hea.exists() and dat.exists()
            and dat.stat().st_size == samples * width):
        return "already complete"
    source = str(PurePosixPath(record).parent / signals[index][0])
    temporary = dat.with_suffix(".dat.part")
    frame_size = count * width
    written = 0
    pending = b""
    with urlopen(BASE + source, timeout=120) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            pending += chunk
            length = len(pending) // frame_size * frame_size
            block, pending = pending[:length], pending[length:]
            # Copy each byte lane to preserve signed integer bit patterns exactly.
            extracted = bytearray(length // count)
            for byte in range(width):
                extracted[byte::width] = block[index * width + byte::frame_size]
            output.write(extracted)
            written += len(extracted)
    if pending or written != samples * width:
        raise ValueError("Incomplete or unexpected signal length; rerun to retry")
    first[1] = "1"
    signal = signals[index][:]
    signal[0] = dat.name
    header = "\n".join([" ".join(first), " ".join(signal)] + lines[count + 1:]) + "\n"
    temporary.replace(dat)
    hea.write_text(header, encoding="ascii")
    return f"saved ABR ({written:,} bytes)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "Physio-ABR-Data")
    parser.add_argument("--limit", type=int, help="Maximum number of raw records to inspect")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = [r.strip() for r in fetch("RECORDS").decode("ascii").splitlines()
               if r.strip().startswith("raw/")]
    if not records:
        raise RuntimeError("No raw records found in the dataset index")
    if args.limit:
        records = records[:args.limit]
    print(f"Destination: {args.output.resolve()}", flush=True)
    failures = []
    for number, record in enumerate(records, 1):
        for attempt in range(3):
            try:
                status = download_record(record, args.output)
                print(f"[{number}/{len(records)}] {record}: {status}", flush=True)
                break
            except Exception as error:
                if attempt == 2:
                    failures.append(record)
                    print(f"FAILED {record}: {error}", file=sys.stderr, flush=True)
                else:
                    time.sleep(2 * (attempt + 1))
    if failures:
        print(f"{len(failures)} records failed. Rerun to retry; complete records are skipped.")
        return 1
    print("Done. Only raw ABR signals and their calibration headers were saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
