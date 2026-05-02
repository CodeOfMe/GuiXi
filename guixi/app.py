"""
Flask web application for GuiXi.

Provides:
- REST API for inference
- WebSocket streaming
- Web-based monitoring dashboard
"""

import asyncio
import json
import threading
from typing import Optional

from flask import Flask, render_template, jsonify, request

from . import __version__
from .api import (
    api_batch_infer,
    api_cache_stats,
    api_clear_cache,
    api_compress,
    api_infer,
    api_stream_infer,
)
from .core import GuiXiServer
from .compress import CompressionMode


app = Flask(__name__, template_folder="templates")
app.config["JSON_SORT_KEYS"] = False


server_instance: Optional[GuiXiServer] = None


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/infer", methods=["POST"])
def infer():
    """Run single inference."""
    data = request.json

    if not data or "prompt" not in data:
        return jsonify({"success": False, "error": "Missing prompt"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(
            api_infer(
                prompt=data["prompt"],
                max_tokens=data.get("max_tokens", 100),
                temperature=data.get("temperature", 0.7),
                compression=data.get("compression", "lz4"),
                cache_policy=data.get("cache", "read"),
            )
        )
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        loop.close()


@app.route("/api/batch", methods=["POST"])
def batch_infer():
    """Run batched inference."""
    data = request.json

    if not data or "prompts" not in data:
        return jsonify({"success": False, "error": "Missing prompts"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(
            api_batch_infer(
                prompts=data["prompts"],
                max_tokens=data.get("max_tokens", 100),
                temperature=data.get("temperature", 0.7),
                compression=data.get("compression", "lz4"),
                batch_size=data.get("batch_size", 10),
            )
        )
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        loop.close()


@app.route("/api/stream", methods=["POST"])
def stream_infer():
    """Stream inference tokens."""
    data = request.json

    if not data or "prompt" not in data:
        return jsonify({"success": False, "error": "Missing prompt"}), 400

    prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 100)
    compression = data.get("compression", "lz4")

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def stream():
                async for result in api_stream_infer(prompt, max_tokens, compression):
                    yield f"data: {json.dumps(result.to_dict())}\n\n"

            gen = stream()
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    return app.response_class(generate(), mimetype="text/event-stream")


@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    """Get cache statistics."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(api_cache_stats())
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        loop.close()


@app.route("/api/cache/clear", methods=["POST"])
def cache_clear():
    """Clear cache."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(api_clear_cache())
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        loop.close()


@app.route("/api/compress", methods=["POST"])
def compress():
    """Compress token data."""
    data = request.json

    if not data or "tokens" not in data:
        return jsonify({"success": False, "error": "Missing tokens"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(
            api_compress(
                data=data["tokens"],
                mode=data.get("mode", "lz4"),
            )
        )
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        loop.close()


@app.route("/api/server/start", methods=["POST"])
def server_start():
    """Start inference server."""
    global server_instance

    if server_instance is not None:
        return jsonify({"success": False, "error": "Server already running"}), 400

    data = request.json or {}
    host = data.get("host", "0.0.0.0")
    port = data.get("port", 8080)
    compression = data.get("compression", "lz4")

    comp_mode = CompressionMode(compression)
    server_instance = GuiXiServer(compression=comp_mode)

    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server_instance.start(host=host, port=port))

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    return jsonify(
        {
            "success": True,
            "data": {"host": host, "port": port},
            "message": f"Server started at {host}:{port}",
        }
    )


@app.route("/api/server/stop", methods=["POST"])
def server_stop():
    """Stop inference server."""
    global server_instance

    if server_instance is None:
        return jsonify({"success": False, "error": "Server not running"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(server_instance.stop())
        server_instance = None
        return jsonify({"success": True, "message": "Server stopped"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        loop.close()


@app.route("/api/server/stats", methods=["GET"])
def server_stats():
    """Get server statistics."""
    if server_instance is None:
        return jsonify({"success": False, "error": "Server not running"}), 400

    return jsonify(
        {
            "success": True,
            "data": server_instance.get_stats(),
        }
    )


@app.route("/api/version", methods=["GET"])
def version():
    """Get version information."""
    return jsonify(
        {
            "success": True,
            "data": {
                "version": __version__,
                "api_version": "1.0",
            },
        }
    )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"success": False, "error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"success": False, "error": "Internal server error"}), 500


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Start the Flask development server."""
    print(f"Starting GuiXi web server at http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/api/version")
    print("Press Ctrl+C to stop")

    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run_server()
