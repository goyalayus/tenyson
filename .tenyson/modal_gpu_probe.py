import os

import modal


app = modal.App("tenyson-gpu-probe")


@app.function(gpu="T4")
def ping() -> str:
    return f"ok:{os.environ.get('MODAL_PROFILE', '')}"


if __name__ == "__main__":
    with app.run():
        print(ping.remote())
