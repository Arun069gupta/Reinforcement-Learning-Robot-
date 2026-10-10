import numpy as np


class RacingTrack:

    def __init__(self):

        # ==========================================================
        # TRACK SIZE
        # ==========================================================

        self.half_length = 6.5
        self.half_width = 4.25

        # Radius of rounded corners
        self.corner_radius = 1.25

        # Number of points used for each curved corner
        self.corner_points = 12

        # Width around centerline considered SAFE
        self.safe_half_width = 0.50

        # Create rounded rectangular centerline
        self.waypoints = self._create_rounded_track()

        self.num_waypoints = len(self.waypoints)

        # Closed loop
        self.closed_waypoints = np.vstack(
            [
                self.waypoints,
                self.waypoints[0]
            ]
        )

        # ==========================================================
        # SEGMENT INFORMATION
        # ==========================================================

        differences = np.diff(
            self.closed_waypoints,
            axis=0
        )

        self.segment_lengths = np.linalg.norm(
            differences,
            axis=1
        )

        self.total_length = np.sum(
            self.segment_lengths
        )

        self.cumulative_lengths = np.concatenate(
            [
                [0.0],
                np.cumsum(self.segment_lengths)
            ]
        )


    # ==============================================================
    # CREATE ROUNDED RECTANGULAR TRACK
    # ==============================================================

    def _create_rounded_track(self):

        points = []

        r = self.corner_radius

        # ----------------------------------------------------------
        # Geometry limits
        # ----------------------------------------------------------

        left = -self.half_length
        right = self.half_length

        bottom = -self.half_width
        top = self.half_width

        # ----------------------------------------------------------
        # Straight-section limits
        # ----------------------------------------------------------

        x_left = left + r
        x_right = right - r

        y_bottom = bottom
        y_top = top

        # ==========================================================
        # BOTTOM STRAIGHT
        # ==========================================================

        bottom_points = np.linspace(
            x_left,
            x_right,
            14
        )

        for x in bottom_points:

            points.append(
                [x, y_bottom]
            )

        # ==========================================================
        # BOTTOM-RIGHT CORNER
        # ==========================================================

        center_x = right - r
        center_y = bottom + r

        angles = np.linspace(
            -np.pi / 2,
            0.0,
            self.corner_points
        )

        for angle in angles[1:]:

            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)

            points.append(
                [x, y]
            )

        # ==========================================================
        # RIGHT STRAIGHT
        # ==========================================================

        right_points = np.linspace(
            bottom + r,
            top - r,
            14
        )

        for y in right_points[1:]:

            points.append(
                [right, y]
            )

        # ==========================================================
        # TOP-RIGHT CORNER
        # ==========================================================

        center_x = right - r
        center_y = top - r

        angles = np.linspace(
            0.0,
            np.pi / 2,
            self.corner_points
        )

        for angle in angles[1:]:

            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)

            points.append(
                [x, y]
            )

        # ==========================================================
        # TOP STRAIGHT
        # ==========================================================

        top_points = np.linspace(
            right - r,
            left + r,
            14
        )

        for x in top_points[1:]:

            points.append(
                [x, top]
            )

        # ==========================================================
        # TOP-LEFT CORNER
        # ==========================================================

        center_x = left + r
        center_y = top - r

        angles = np.linspace(
            np.pi / 2,
            np.pi,
            self.corner_points
        )

        for angle in angles[1:]:

            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)

            points.append(
                [x, y]
            )

        # ==========================================================
        # LEFT STRAIGHT
        # ==========================================================

        left_points = np.linspace(
            top - r,
            bottom + r,
            14
        )

        for y in left_points[1:]:

            points.append(
                [left, y]
            )

        # ==========================================================
        # BOTTOM-LEFT CORNER
        # ==========================================================

        center_x = left + r
        center_y = bottom + r

        angles = np.linspace(
            np.pi,
            3 * np.pi / 2,
            self.corner_points
        )

        for angle in angles[1:]:

            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)

            points.append(
                [x, y]
            )

        return np.array(
            points,
            dtype=np.float32
        )


    # ==============================================================
    # GET TRACK INFORMATION
    # ==============================================================

    def get_track_info(self, position):

        position = np.asarray(
            position,
            dtype=np.float32
        )

        best_distance = float("inf")

        best_progress_distance = 0.0

        best_segment = 0

        # ----------------------------------------------------------
        # Find closest point on centerline
        # ----------------------------------------------------------

        for i in range(self.num_waypoints):

            start = self.closed_waypoints[i]

            end = self.closed_waypoints[i + 1]

            segment = end - start

            segment_length_squared = np.dot(
                segment,
                segment
            )

            if segment_length_squared == 0:

                continue

            t = np.dot(
                position - start,
                segment
            ) / segment_length_squared

            t = np.clip(
                t,
                0.0,
                1.0
            )

            closest_point = (
                start +
                t * segment
            )

            distance = np.linalg.norm(
                position - closest_point
            )

            if distance < best_distance:

                best_distance = distance

                best_segment = i

                best_progress_distance = (
                    self.cumulative_lengths[i]
                    +
                    t * self.segment_lengths[i]
                )

        # ----------------------------------------------------------
        # Progress
        # ----------------------------------------------------------

        progress = (
            best_progress_distance /
            self.total_length
        )

        return (
            progress,
            best_segment,
            best_distance
        )


    # ==============================================================
    # NEXT WAYPOINT
    # ==============================================================

    def get_next_waypoint(self, segment):

        next_index = (
            segment + 1
        ) % self.num_waypoints

        return self.waypoints[next_index]


    # ==============================================================
    # CENTERLINE POINT
    # ==============================================================

    def get_centerline_point(self, segment):

        segment = segment % self.num_waypoints

        return self.waypoints[segment].copy()


    # ==============================================================
    # TRACK DIRECTION
    # ==============================================================

    def get_track_direction(self, segment):

        segment = segment % self.num_waypoints

        start = self.closed_waypoints[segment]

        end = self.closed_waypoints[segment + 1]

        direction = end - start

        magnitude = np.linalg.norm(
            direction
        )

        if magnitude < 1e-8:

            return np.array(
                [1.0, 0.0],
                dtype=np.float32
            )

        return (
            direction / magnitude
        ).astype(np.float32)


    # ==============================================================
    # SAFE CORRIDOR CHECK
    # ==============================================================

    def is_inside_safe_line(self, distance):

        return (
            distance <= self.safe_half_width
        )


    # ==============================================================
    # DISTANCE OUTSIDE SAFE LINE
    # ==============================================================

    def get_distance_outside_safe_line(
        self,
        distance
    ):

        return max(
            0.0,
            distance - self.safe_half_width
        )