import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CustomHandler(SimpleHTTPRequestHandler):
    """Class to Handle HTTP Requests"""

    def do_GET(self):  # noqa: N802
        """Function that finds JSONS in current folder"""
        if self.path == "/list_jsons":  # Endpoint to list JSON files
            json_files = [f for f in os.listdir() if f.endswith(".json")]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(json_files).encode())
        else:
            super().do_GET()  # Serve static files (HTML, JS, CSS, etc.)


if __name__ == "__main__":
    port = 8181  # Match your existing setup
    server_address = ("", port)
    httpd = HTTPServer(server_address, CustomHandler)
    print(f"3D Visualization Server is running on http://localhost:{port}")
    httpd.serve_forever()
