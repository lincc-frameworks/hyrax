# Service Initialization
To start the interactive server

`python start_3d_viz_server.py`

This will launch the service on the 8181 port. 


If you are running this on a remote machine, forward this port appropriately using something like

`ssh -N -L 8181:server_name:8181 username@loginnode.com`


Finally, navigate to http://localhost:8181/



## Saving UMAPs as json
To convert a UMAP created by fibad to the JSON format, use save_umap_to_json.py


## Simpler Notebook Version
For a more straightforward plotly 3d plot, use the function in plotly_3d.py
