# Server Initialization
To start the interactive server, type

`python start_3d_viz_server.py`

This will launch the service on the 8181 port. If you are running this on a remote machine, forward this port appropriately using something like

`ssh -N -L 8181:server_name:8181 username@loginnode.com`

Finally, navigate to http://localhost:8181/

You can also optionally change the port the server is being displayed on; and also pass a folder containing your cutouts. Note that the path passed to `cutouts_dir` is relative to the location of root of the server (i.e., location of the `start_3d_viz_server.py` file)

To see all the command line arguments, do `python start_3d_viz_server.py --help`

## FAQs
1. If there are repeated IDs in your dataset, you will see the second instance of the object not loaded in the image viewer. Instead you will keep seeing the image loading spinning wheel symbol.

# Saving UMAPs as json
To convert a UMAP created by fibad to the JSON format, use save_umap_to_json.py
This can be run using `python save_umap_to_json.py /path/to/results/dir`
To see optional argments do `python save_umap_to_json.py --help`


# Simpler Notebook Version
For a more straightforward plotly 3d plot, use the function in plotly_3d.py
