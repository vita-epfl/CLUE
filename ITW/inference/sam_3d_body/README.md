SAM 3D Body (body branch)
=========================

The model code of SAM 3D Body (https://github.com/facebookresearch/sam-3d-body, Meta), under the SAM License (LICENSE),
reduced to the image-to-full-body MHR regression path this code uses: no demo script, detector, segmentor, FOV estimator
or visualization stack, and no second-stage hand decoder. The MHR head can predict hands and face identity
(``head_pose.predict_hands_face``); the CLUE inference code turns this on.
