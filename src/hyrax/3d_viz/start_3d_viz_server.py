import argparse
import json
import os
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


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
        elif self.path.startswith("/convert_tensor/"):  # New endpoint to convert tensor files
            self._handle_tensor_conversion()
        else:
            # print(f"DEBUG: Attempting to serve file from: {os.path.join(os.getcwd(), self.path[1:])}")
            super().do_GET()  # Serve static files (HTML, JS, CSS, etc.)

    def _handle_tensor_conversion(self):
        """Handle tensor file conversion to JSON format for JavaScript processing"""
        try:
            # Extract filename from URL path
            # URL format: /convert_tensor/path/to/file.pt
            tensor_path = self.path[len("/convert_tensor/") :]
            tensor_path = urllib.parse.unquote(tensor_path)  # Decode URL encoding

            # Construct full path
            full_path = Path(tensor_path)
            if not full_path.is_absolute():
                full_path = Path(self.cutouts_dir) / tensor_path

            print(f"DEBUG: Converting tensor file: {full_path}")

            if not full_path.exists():
                self._send_error(404, f"Tensor file not found: {full_path}")
                return

            if full_path.suffix.lower() != ".pt":
                self._send_error(400, "File is not a PyTorch tensor (.pt)")
                return

            # Convert tensor to JSON format that matches what JavaScript expects
            tensor_data = self._convert_tensor_to_json(full_path)

            # Send JSON response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")  # Allow CORS
            self.end_headers()
            self.wfile.write(json.dumps(tensor_data).encode())

        except Exception as e:
            print(f"ERROR: Failed to convert tensor: {e}")
            self._send_error(500, f"Failed to convert tensor: {str(e)}")

    def _convert_tensor_to_json(self, tensor_path):
        """
        Convert a PyTorch tensor file to JSON format that JavaScript can process
        similar to how FITS data is processed
        """
        try:
            import torch

            # Load the tensor
            tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)

            # Extract band 3 (i-band equivalent, matching visualize.py:571)
            if len(tensor.shape) >= 2:
                if len(tensor.shape) == 3 and tensor.shape[0] > 3:
                    # Multi-channel tensor, extract band 3 (index 3)
                    image_data = tensor[3].numpy()
                elif len(tensor.shape) == 3:
                    # Use first available channel if less than 4 channels
                    image_data = tensor[0].numpy()
                else:
                    # 2D tensor, use as-is
                    image_data = tensor.numpy()
            else:
                raise ValueError(f"Tensor has unsupported shape: {tensor.shape}")

            # Get dimensions
            height, width = image_data.shape

            # Convert to list for JSON serialization
            # Keep as 2D structure for easier processing in JavaScript
            data_nested = image_data.tolist()

            return {
                "width": int(width),
                "height": int(height),
                "data": data_nested,  # 2D array structure
                "type": "tensor",
                "format": "2d_array",
            }

        except ImportError as e:
            raise RuntimeError("PyTorch is required for tensor conversion but not available") from e
        except Exception as e:
            raise RuntimeError(f"Failed to process tensor file: {str(e)}") from e

    def _send_error(self, code, message):
        """Send an error response"""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        error_data = {"error": message}
        self.wfile.write(json.dumps(error_data).encode())


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
