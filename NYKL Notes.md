conda activate xoftr

python run_XoFTR_on_AftrImgPairs.py --port 12676 --show --calib calibration_aftr.yaml --top_k 15


❯  Let's pretend the first/left image (imageA) is taken from a sensor that is approaching the aircraft (think of automated aerial refueling). We are trying to figure out the pose of the camera that acquired imageA. Now, let's assume we know the full dimensions of the 3D aircraft that is being approached. Let's also assume that we know exactly where the camera's pose was taken in imageB. Given we know the pose of the camera that acquired imageB AND we know the exact scale and dimensions of the aircraft, what can we do to compute the translation between imageA and imageB? We already have the rotation R and the direction T, we sort of just need to compute an approximate scale or come up with another method to find T. Thoughts?

Great question — you're right that the Essential matrix only gives translation direction, not magnitude. But you have two strong priors that can recover scale:

Option 1: PnP with known 3D points (best option)

Since you know the aircraft's 3D model and imageB's camera pose, you can:
1. Use XoFTR matches to find correspondences between imageA and imageB
2. Since imageB's pose is known, back-project imageB's matched 2D keypoints into 3D using the aircraft model (ray-casting or known feature locations on the aircraft)
3. Now you have 3D-to-2D correspondences for imageA
4. Solve with cv2.solvePnP — this gives you imageA's full pose (R + scaled T) directly, bypassing the Essential matrix entirely

This is the cleanest path. PnP gives you absolute position in the aircraft's coordinate frame with real-world scale. You don't need the Essential matrix decomposition at all — the XoFTR matches are the input, but the geometry solver changes from E-matrix to PnP.

Option 2: Scale from known geometry

Keep the Essential matrix approach but recover scale using known distances:
1. Pick two matched points on the aircraft whose real-world distance you know (e.g., wingtip to wingtip, or two refueling receptacle features)
2. Triangulate them using R and unit-T to get 3D positions (up to scale)
3. Compare the triangulated distance to the known distance — the ratio gives you the scale factor for T

Simpler to implement but less robust — depends on picking good point pairs.

My recommendation: Option 1 (PnP) is what you'd use in a real AAR system. It directly gives you imageA's pose in the tanker's reference frame. The prerequisite is being able to map imageB keypoints to 3D aircraft coordinates — how are you planning to establish that? Do you have a textured 3D mesh of the aircraft, or known fiducial/feature locations?