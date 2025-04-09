import cv2 as cv
import numpy as np

video_path = 'C:/Users/Hi/Desktop/Chessboard.mp4'

board_size = (8, 6)
board_cellsize = 25.0 

K = np.array([[1000, 0, 320],
              [0, 1000, 240],
              [0, 0, 1]], dtype=np.float32)
dist_coeff = np.zeros(5)

obj_points = board_cellsize * np.array(
    [[c, r, 0] for r in range(board_size[1]) for c in range(board_size[0])],
    dtype=np.float32
)

x0, x1 = 4, 6
y0, y1 = 2, 4
z_bottom = 0
z_top = -5

lower_square = board_cellsize * np.array([
    [x0, y0, z_bottom],
    [x1, y0, z_bottom],
    [x1, y1, z_bottom],
    [x0, y1, z_bottom]
], dtype=np.float32)

upper_square = board_cellsize * np.array([
    [x0, y0, z_top],
    [x1, y0, z_top],
    [x1, y1, z_top],
    [x0, y1, z_top]
], dtype=np.float32)

center_point = board_cellsize * np.array(
    [[(x0 + x1) / 2, (y0 + y1) / 2, (z_bottom + z_top) / 2]],
    dtype=np.float32
)

hourglass_points = np.vstack([lower_square, upper_square, center_point]) 

cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video cannot be opened.")
    exit()

while True:
    valid, frame = cap.read()
    if not valid:
        break

    frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, board_size,
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        success, rvec, tvec = cv.solvePnP(obj_points, corners, K, dist_coeff)

        if success:
            imgpts, _ = cv.projectPoints(hourglass_points, rvec, tvec, K, dist_coeff)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            lower = imgpts[0:4]
            upper = imgpts[4:8]
            center = imgpts[8]

            cv.polylines(frame, [lower], isClosed=True, color=(255, 0, 0), thickness=2)
            cv.polylines(frame, [upper], isClosed=True, color=(0, 255, 0), thickness=2)

            for pt in lower:
                cv.line(frame, tuple(pt), tuple(center), (200, 100, 255), 2)

            for pt in upper:
                cv.line(frame, tuple(pt), tuple(center), (100, 255, 255), 2)

    cv.imshow("3D Hourglass AR", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()