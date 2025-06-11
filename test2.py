from gps3 import gps3

# Initialize the GPS connection
gps_socket = gps3.GPSDSocket()
data_stream = gps3.DataStream()
gps_socket.connect()
gps_socket.watch()

for new_data in gps_socket:
    if new_data:
        data_stream.unpack(new_data)
        # Extract latitude and longitude
        latitude = data_stream.TPV['lat']
        longitude = data_stream.TPV['lon']
        if latitude and longitude:  # Check if values are not None
            print(f"Latitude: {latitude}, Longitude: {longitude}")
            break