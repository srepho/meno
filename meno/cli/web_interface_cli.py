"""Command-line interface for the Meno web interface."""

import argparse
import sys
from meno.web_interface import launch_web_interface


def main():
    """Run the Meno web interface from the command line."""
    parser = argparse.ArgumentParser(
        description="Launch the Meno topic modeling web interface."
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8050,
        help="Port to run the web server on (default: 8050)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Launching Meno web interface on port {args.port}...")
        print("Use Ctrl+C to stop the server")
        launch_web_interface(port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()