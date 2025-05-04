import argparse
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CustomHandler(SimpleHTTPRequestHandler):
    """Class to Handle HTTP Requests"""

    def do_GET(self):  # noqa: N802
        """Function that finds JSONS in current folder"""
        # print(f"DEBUG: Requested path: {self.path}")

        if self.path == "/list_jsons":  # Endpoint to list JSON files
            json_files = [f for f in os.listdir() if f.endswith(".json")]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(json_files).encode())
        elif self.path == "/get_cutouts_dir":  # New endpoint to get cutouts directory
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"cutouts_dir": self.cutouts_dir}).encode())
        else:
            # print(f"DEBUG: Attempting to serve file from: {os.path.join(os.getcwd(), self.path[1:])}")
            super().do_GET()  # Serve static files (HTML, JS, CSS, etc.)


def main():
    """
    Entry point for the 3D UMAP Visualization Server.

    This function:
    1. Parses command line arguments for the cutouts directory and port number
    2. Validates the cutouts directory existence and creates it if necessary
    3. Sets up an HTTP server with the CustomHandler to:
       - Serve static files (HTML, CSS, JS)
       - Provide a '/list_jsons' endpoint to enumerate available datasets
    4. Starts the server and runs indefinitely until interrupted

    Command-line Arguments:
        --cutouts-dir: Directory containing FITS image cutouts (default: ./cutouts)
                       This path is relative to the location of this file
        --port: Port to run the server on (default: 8181)

    Usage Example:
        python start_3d_viz_server.py --cutouts-dir /path/to/images --port 8080
    """

    parser = argparse.ArgumentParser(description="Start 3D Visualization Server")
    parser.add_argument(
        "--cutouts-dir",
        default="cutouts",
        help="Directory containing FITS image cutouts (default: ./cutouts). This Path is relative to\
            the location of this script.",
    )
    parser.add_argument("--port", type=int, default=8181, help="Port to run the server on (default: 8181)")

    args = parser.parse_args()

    # Store the cutouts directory in an environment variable
    os.environ["CUTOUTS_DIR"] = args.cutouts_dir

    # Create a global variable to access in the handler
    CustomHandler.cutouts_dir = args.cutouts_dir

    # Verify that cutouts directory exists
    if not os.path.isdir(args.cutouts_dir):
        print(f"Warning: Cutouts directory '{args.cutouts_dir}' not found.")
        print(f"Will create directory '{args.cutouts_dir}' if images are requested.")
        os.makedirs(args.cutouts_dir, exist_ok=True)

    server_address = ("", args.port)
    httpd = HTTPServer(server_address, CustomHandler)
    print(f"3D Visualization Server is running on http://localhost:{args.port}")
    print(f"Using cutouts directory: {args.cutouts_dir}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
