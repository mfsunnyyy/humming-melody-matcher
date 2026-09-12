import argparse
import json

from live_capture import capture_audio
from humming_matcher import identify_humming_live


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=float, default=10.0)
    parser.add_argument("--db", type=str, default="database/song_index.db")
    parser.add_argument("--log", type=str, default="humming_query_log.json")
    parser.add_argument(
        "--query-type",
        choices=["humming", "cover"],
        default="cover",
    )
    return parser.parse_args()


def ascii_safe(value: str) -> str:
    try:
        return value.encode("ascii", "replace").decode("ascii")
    except Exception:
        return str(value)


if __name__ == "__main__":
    args = parse_args()

    print(
        f"Melody recognition: recording {args.record:.1f}s "
        f"as {args.query_type}"
    )

    audio, sr = capture_audio(duration=args.record)

    winner, top, info = identify_humming_live(
        audio,
        sr=sr,
        db_path=args.db,
        top_k=5,
        query_type=args.query_type,
    )

    if winner:
        print(f"\nDETECTED ({args.query_type}):", ascii_safe(winner))
    else:
        print("\nNO CONFIDENT MELODY MATCH")

    print("Top candidates:")
    for rank, candidate in enumerate(top, start=1):
        print(
            f"  {rank}. {ascii_safe(candidate['song_name'])} "
            f"distance={candidate['distance']:.5f}"
        )

    print("Info:", info)

    data = {
        "winner": winner,
        "query_type": args.query_type,
        "record_seconds": args.record,
        "sample_rate": sr,
        "top": top,
        "info": info,
    }

    with open(args.log, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Diagnostics written to {args.log}")
