from environments.track import RacingTrack


track = RacingTrack()


test_positions = [

    [-5.0, -3.25],
    [0.0, -3.25],
    [5.0, -3.25],
    [5.0, 0.0],
    [5.0, 3.25],
    [0.0, 3.25],
    [-5.0, 3.25],
    [-5.0, 0.0],

]


for position in test_positions:

    progress, waypoint, distance = track.get_progress(
        position
    )

    print(
        f"Position: {position} | "
        f"Waypoint: {waypoint} | "
        f"Progress: {progress:.3f} | "
        f"Distance: {distance:.3f}"
    )
