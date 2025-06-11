import gps

# Create a GPS session
session = gps.gps(mode=gps.WATCH_ENABLE)

# Read GPS data
for report in session:
    if report['class'] == 'TPV':  # Check for time-position-velocity reports
        if hasattr(report, 'lat') and hasattr(report, 'lon'):
            latitude = report.lat
            longitude = report.lon
            print(f"Latitude: {latitude}, Longitude: {longitude}")
            break