import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from server.app import create_app


def main():
    parser = argparse.ArgumentParser(description="MOSS meeting-service public server")
    parser.add_argument("--host", default=os.environ.get("MTD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MTD_PORT", "8000")))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--worker-key", default="", help="worker API key (auto-generated if empty)")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["MTD_DATA_DIR"] = args.data_dir
    if args.worker_key:
        os.environ["MTD_WORKER_KEY"] = args.worker_key

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
