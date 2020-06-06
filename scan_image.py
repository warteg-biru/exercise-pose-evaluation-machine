# Scan image for keypoints
def scan_image(img_path):
model_path = './deep_sort/model_data/mars-small128.pb'
params = set_params()
nms_max_overlap = 1.0

# Constructing OpenPose object allocates GPU memory
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Set tracker
max_cosine_distance = 0.2
nn_budget = 100
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Initialize encoder
encoder = create_box_encoder(model_path, batch_size=1)

# Opening image in OpenCV
imageToProcess = cv2.imread(img_path)

# Get data points (datum)
datum = op.Datum()
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum])

# Display the stream
output_image = datum.cvOutputData
arr = []
boxes = []

# Loop each of the 17 keypoints
for keypoint in datum.poseKeypoints:
    pop_all(arr)
    x_high = 0
    x_low = 9999
    y_high = 0
    y_low = 9999

    # Get highest and lowest keypoints
    for count, x in enumerate(keypoint):
        # Avoid x=0 and y=0 (because some keypoints that are not discovered result as x=0 and y=0)
        if x[0] != 0 and x[1] != 0:
            if x_high < x[0]:
                x_high = x[0]
            if x_low > x[0]:
                x_low = x[0]
            if y_high < x[1]:
                y_high = x[1]
            if y_low > x[1]:
                y_low = x[1]

        # Append keypoints 0, 1, or 15 into the array
        if count == 0 or count == 15 or count == 1:
            kp1 = x[0]
            kp2 = x[1]
            kp3 = x[2]
            KP = cv2.KeyPoint(kp1, kp2, kp3)
            arr.append(KP)

    # Find the highest and lowest position of x and y (Used to draw rectangle)
    if y_high - y_low > x_high - x_low:
        height = y_high-y_low
        width = x_high-x_low
    else:
        height = x_high-x_low
        width = y_high-y_low

    # Draw rectangle (get width and height)
    y_high = int(y_high + height / 40)
    y_low = int(y_low - height / 12)
    x_high = int(x_high + width / 5)
    x_low = int(x_low - width / 5)

    # Normalize keypoint
    normalized_keypoints = normalize_keypoints(keypoint, x_low, y_low)
    insert_array_to_db(normalized_keypoints)

    # Make the box
    boxes.append([x_low, y_low, width, height])

    # Encode the features inside the designated box
    features = encoder(output_image, boxes)

    # For a non-empty item add to the detection array
    def nonempty(xywh): return xywh[2] != 0 and xywh[3] != 0
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(
        boxes, features) if nonempty(bbox)]

    # Run non-maxima suppression.
    np_boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        np_boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Update tracker.
    tracker.predict()
    tracker.update(detections)

    # Draw rectangle and put text
    for track in tracker.tracks:
        bbox = track.to_tlwh()
        cv2.rectangle(output_image, (x_high, y_high),
                        (x_low, y_low), (255, 0, 0), 2)
        cv2.putText(output_image, "id%s - ts%s" % (track.track_id, track.time_since_update),
                    (int(bbox[0]), int(bbox[1])-20), 0, 5e-3 * 100, (0, 255, 0), 2)

# Output image
# cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", output_image)
# cv2.waitKey(0)