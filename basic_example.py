import os
from flask import Flask

print("--- STARTING HELLO WORLD TEST ---", flush=True)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "alive", "mode": "test"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4768))
    print(f"--- LISTENING ON PORT {port} ---", flush=True)

    app.run(host="0.0.0.0", port=port)
