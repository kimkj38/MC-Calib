#include "opencv2/core/core.hpp"
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdio.h>

#include "Calibration.hpp"
#include "logger.h"
#include "point_refinement.h"

/**
 * @brief Initialize the number of cameras and the 3D Boards
 *
 * @param config_path path to the configuration file
 */
Calibration::Calibration(const std::string config_path) {
  cv::FileStorage fs; // cv::FileStorage to read calibration params from file
  int distortion_model;
  std::vector<int> distortion_per_camera;
  std::vector<int> boards_index;
  int nb_x_square, nb_y_square;
  float length_square, length_marker;
  const bool is_file_available =
      boost::filesystem::exists(config_path) && config_path.length() > 0;
  if (!is_file_available) {
    LOG_FATAL << "Config path '" << config_path << "' doesn't exist.";
    return;
  }
  fs.open(config_path, cv::FileStorage::READ);
  fs["number_camera"] >> nb_camera_; //카메라의 수
  fs["number_board"] >> nb_board_; //board의 수
  fs["refine_corner"] >> refine_corner_; //corner refinement 여부
  fs["min_perc_pts"] >> min_perc_pts_; //detection을 하기 위해 보여야 하는 point들의 최소 percentage
  fs["number_x_square"] >> nb_x_square; //x축 방향으로 square의 수
  fs["number_y_square"] >> nb_y_square; //y축 방향으로 sqaure의 수
  fs["root_path"] >> root_dir_; //root path
  fs["cam_prefix"] >> cam_prefix_; //"Cam_"
  fs["ransac_threshold"] >> ransac_thresh_; //ransac threshold
  fs["number_iterations"] >> nb_iterations_; //non linear refinement를 위한 최대 반복 수
  fs["distortion_model"] >> distortion_model; //0:Brown(perspective), 1:Kannala(fisheye)
  fs["distortion_per_camera"] >> distortion_per_camera; //카메라 별 distortion model 명시. 전부 같은 모델 사용하면 []
  fs["boards_index"] >> boards_index; //[5,10]으로 두면 index 5와 10을 가진 32개의 board. []로 두면 0부터 board의 수만큼 인덱싱 
  fs["length_square"] >> length_square; //length square
  fs["length_marker"] >> length_marker; //length marker
  fs["save_path"] >> save_path_; //저장할 경로
  fs["camera_params_file_name"] >> camera_params_file_name_; //저장할 output의 이름
  fs["cam_params_path"] >> cam_params_path_; //initial로 사용할 intrinsic parameter 경로. 없으면 "None"
  fs["save_reprojection"] >> save_repro_; 
  fs["save_detection"] >> save_detect_;
  fs["square_size_per_board"] >> square_size_per_board_; //board별 square의 사이즈. 전부 동일하면 []
  fs["number_x_square_per_board"] >> number_x_square_per_board_; //board별 x축 방향 square의 수. 전부 동일하면 []
  fs["number_y_square_per_board"] >> number_y_square_per_board_; //board별 y축 방향 square의 수. 전부 동일하면 []
  fs["resolution_x_per_board"] >> resolution_x_per_board_; //x축 방향 pixel 수
  fs["resolution_y_per_board"] >> resolution_y_per_board_; //y축 방향 pixel 수
  fs["he_approach"] >> he_approach_; //0: bootstrapped he technique, 1: traditional he
  fs["fix_intrinsic"] >> fix_intrinsic_; //intial intrinsic이 있을 때 1로 설정하면 intrinsic 고정

  fs.release(); // close the input file

  // Check if multi-size boards are used or not
  // multiple board 사용 시 config 파일에서 정의한 파라미터들 다시 확인
  if (boards_index.size() != 0) {
    nb_board_ = boards_index.size();
  }
  int max_board_idx = nb_board_ - 1; //board의 max index
  if (boards_index.size() != 0) {
    max_board_idx = *max_element(boards_index.begin(), boards_index.end());
  }
  if (square_size_per_board_.size() == 0) {
    number_x_square_per_board_.assign(max_board_idx + 1, nb_x_square);
    number_y_square_per_board_.assign(max_board_idx + 1, nb_y_square);
  }

  LOG_INFO << "Nb of cameras : " << nb_camera_
           << "   Nb of Boards : " << nb_board_
           << "   Refined Corners : " << refine_corner_
           << "   Distortion mode : " << distortion_model;

  // check if the save dir exist and create it if it does not
  // output 저장할 디렉토리 확인
  if (!boost::filesystem::exists(save_path_) && save_path_.length() > 0) {
    boost::filesystem::create_directories(save_path_);
  }

  // prepare the distortion type per camera
  // distortion_per_camera가 []인 경우 전부 같은 모델로 할당
  if (distortion_per_camera.size() == 0)
    distortion_per_camera.assign(nb_camera_, distortion_model);

  // Initialize Cameras
  // cams_[i]에 distortion model을 가진 camera 정의
  for (int i = 0; i < nb_camera_; i++) {
    std::shared_ptr<Camera> new_cam =
        std::make_shared<Camera>(i, distortion_per_camera[i]);
    cams_[i] = new_cam;
  }

  // Prepare the charuco patterns
  // boards_index가 []일 때 board의 수만큼 크기 할당 후 인덱스 부여
  if (boards_index.size() == 0) {
    boards_index.resize(nb_board_);
    std::iota(boards_index.begin(), boards_index.end(), 0);
  }

  std::map<int, cv::Ptr<cv::aruco::CharucoBoard>> charuco_boards; //cpp의 map은 dictionary 구조. map(key, value)
  int offset_count = 0;
  //board의 개수만큼 반복문
  for (int i = 0; i <= max_board_idx; i++) {
    //create(x축 방향 square의 수, y축 방향 square의 수, square의 가로 길이, marker의 가로 길이, marker의 type에 대한 dictionary). square와 marker의 길이는 일반적으로 meter 단위
    cv::Ptr<cv::aruco::CharucoBoard> charuco = cv::aruco::CharucoBoard::create(
        number_x_square_per_board_[i], number_y_square_per_board_[i], 
        length_square, length_marker, dict_);

    //id는 charuco board의 개수에 따른 index가 아니라 marker의 개수에 따라 할당? 어디에 쓰는건지 모르겠다.
    if (i != 0) // If it is the first board then just use the standard idx
    {
      int id_offset = charuco_boards[i - 1]->ids.size() + offset_count;
      offset_count = id_offset;
      for (auto &id : charuco->ids)
        id += id_offset;
    }
    charuco_boards[i] = charuco;
  }

  // Initialize the 3D boards
  for (int i = 0; i < nb_board_; i++) {
    // Initialize board
    std::shared_ptr<Board> new_board = std::make_shared<Board>(config_path, i);
    boards_3d_[i] = new_board;
    //point의 개수
    boards_3d_[i]->nb_pts_ =
        (boards_3d_[i]->nb_x_square_ - 1) * (boards_3d_[i]->nb_y_square_ - 1);
    
    //3d board와 매칭되는 charuco board
    boards_3d_[i]->charuco_board_ = charuco_boards[boards_index[i]];
    
    //point의 개수만큼 pts_3d에 할당
    boards_3d_[i]->pts_3d_.reserve(boards_3d_[i]->nb_pts_);
    
    // Prepare the 3D pts of the boards
    // x,y축 square의 개수만큼 이중 반복문 돌면서 (square size * x축 index, square size * y축 index, 0)으로 3d point 할당
    for (int y = 0; y < boards_3d_[i]->nb_y_square_ - 1; y++) {
      for (int x = 0; x < boards_3d_[i]->nb_x_square_ - 1; x++) {
        boards_3d_[i]->pts_3d_.emplace_back(x * boards_3d_[i]->square_size_,
                                            y * boards_3d_[i]->square_size_, 0);
      }
    }
  }
}

/**
 * @brief Extract necessary boards info from initialized paths
 *
 */
void Calibration::boardExtraction() {
  std::unordered_set<cv::String> allowed_exts = {"jpg",  "png", "bmp",
                                                 "jpeg", "jp2", "tiff"};

  // iterate through the cameras
  // 카메라 수만큼 반복문
  for (int cam = 0; cam < nb_camera_; cam++) {
    // prepare the folder's name
    
    //setw(n): 출력에 사용할 필드의 널이
    //setfill(c): 빈 공간을 c로 채운다
    //0001, 0002과 같이 cam에 id를 부여
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << cam + 1;
    std::string cam_nb = ss.str(); //cam_nb가 id 역할
    std::string cam_path = root_dir_ + cam_prefix_ + cam_nb; //cam의 경로. cam_prefix는 "Cam_"(config 파일에서 설정) 
    LOG_INFO << "Extraction camera " << cam_nb;

    // iterate through the images for corner extraction
    std::vector<cv::String> fn;
    cv::glob(cam_path + "/*.*", fn, true); // glob(찾을 파일 경로, 찾은 파일 경로, recursive(true or false)

    // filter based on allowed extensions
    // 위에서 정의한 확장자만 필터링
    std::vector<cv::String> fn_filtered;
    for (cv::String cur_path : fn) {
      std::size_t ext_idx = cur_path.find_last_of(".");
      cv::String cur_ext = cur_path.substr(ext_idx + 1);
      if (allowed_exts.find(cur_ext) != allowed_exts.end()) {
        fn_filtered.push_back(cur_path);
      }
    }
    fn = fn_filtered;

    // 해당 카메라가 가지고 있는 이미지의 수 = frame의 수
    size_t count_frame =
        fn.size(); // number of allowed image files in images folder
    
    // frame 수만큼 반복문 돌면서 frame path 및 이미지를 matrix 형태로 저장(frame_path, currentIm)
    for (size_t frameind = 0; frameind < count_frame; frameind = frameind + 1) {
      // open Image
      cv::Mat currentIm = cv::imread(fn[frameind]);
      std::string frame_path = fn[frameind];
      // detect the checkerboard on this image
      LOG_DEBUG << "Frame index :: " << frameind;
      
      //이미지에서 board detection
      detectBoards(currentIm, cam, frameind, frame_path);
      LOG_DEBUG << frameind;
      // displayBoards(currentIm, cam, frameind); // Display frame
    }
  }
}

/**
 * @brief Detect boards on an image
 *
 * @param image Image on which we would like to detect the board
 * @param cam_idx camera index which acquire the frame
 * @param frame_idx frame index
 */
void Calibration::detectBoards(const cv::Mat image, const int cam_idx,
                               const int frame_idx,
                               const std::string frame_path) {
  // Greyscale image for subpixel refinement
  cv::Mat graymat;
  cv::cvtColor(image, graymat, cv::COLOR_BGR2GRAY); //이미지를 Grayscale로 바꿔준다

  // Initialize image size
  // 해당 이미지의 cols와 rows
  cams_[cam_idx]->im_cols_ = graymat.cols;
  cams_[cam_idx]->im_rows_ = graymat.rows;

  // Datastructure to save the checkerboard corners
  // board 및 marker의 corner에 대한 id와 2d points
  std::map<int, std::vector<int>>
      marker_idx; // key == board id, value == markersIDs on MARKERS markerIds
  std::map<int, std::vector<std::vector<cv::Point2f>>>
      marker_corners; // key == board id, value == 2d points visualized on
                      // MARKERS
  std::map<int, std::vector<cv::Point2f>>
      charuco_corners; // key == board id, value == 2d points on checkerboard
  std::map<int, std::vector<int>>
      charuco_idx; // key == board id, value == ID corners on checkerboard

  charuco_params_->adaptiveThreshConstant = 1;

  // board의 수만큼 반복문
  for (int i = 0; i < nb_board_; i++) {
    // cv::aruco::detectMarkers( 마커를 찾을 이미지, marker의 type이 담긴 dictionary, marker의 corner 리스트(Nx4), marker의 id 리스트, DetectionParametes)
    cv::aruco::detectMarkers(image, boards_3d_[i]->charuco_board_->dictionary, 
                             marker_corners[i], marker_idx[i],
                             charuco_params_); // detect markers

    if (marker_corners[i].size() > 0) {
      // cv::aruco::interpolateCornersCharuco(marker의 corners 리스트, marker의 id 리스트, 이미지, board의 layout, board의 corner 리스트, board의 id 리스트, camera matrix, distortion 계수)
      cv::aruco::interpolateCornersCharuco(marker_corners[i], marker_idx[i],
                                           image, boards_3d_[i]->charuco_board_,
                                           charuco_corners[i], charuco_idx[i]);
    }

    // board의 corner의 개수가 thereshold 이상일 때만 refinement(논문1)
    // boards are discarded from further consideration
    if (charuco_corners[i].size() >
        (int)round(min_perc_pts_ * boards_3d_[i]->nb_pts_)) {
      LOG_INFO << "Number of detected corners :: " << charuco_corners[i].size();
      // Refine the detected corners
      if (refine_corner_ == true) {
        std::vector<SaddlePoint> refined;
        saddleSubpixelRefinement(graymat, charuco_corners[i], refined,
                                 corner_ref_window_, corner_ref_max_iter_);
        for (int j = 0; j < charuco_corners[i].size(); j++) {
          //refinement의 x 혹은 y 좌표가 무한대면 break
          if (std::isinf(refined[j].x) || std::isinf(refined[j].y)) {
            break;
          }
          // refinement한 값으로 업데이트
          charuco_corners[i][j].x = refined[j].x;
          charuco_corners[i][j].y = refined[j].y;
        }
      }

      // Check for colinnerarity(논문 1)
      // 독립변수가 다른 독립변수들과 선형 독립이 아닌 경우를 의미
      // 회귀 분석시 변수들 간의 관계가 클 경우 부정적인 영향을 미친다.
      std::vector<cv::Point2f> pts_on_board_2d;
      pts_on_board_2d.reserve(charuco_idx[i].size()); //board의 corner 개수만큼 공간 할당

      // board의 corner 개수만큼 반복문 돌며 3d point의 (x,y)를 pts_on_board_2d에 저장
      for (const auto &charuco_idx_at_board_id : charuco_idx[i]) {
        pts_on_board_2d.emplace_back(
            boards_3d_[i]->pts_3d_[charuco_idx_at_board_id].x,
            boards_3d_[i]->pts_3d_[charuco_idx_at_board_id].y);
      }
      double dum_a, dum_b, dum_c;
      double residual;
      //residual을 계산
      calcLinePara(pts_on_board_2d, dum_a, dum_b, dum_c, residual);

      // Add the board to the datastructures (if it passes the collinearity
      // check)
      // residual이 square size의 0.1보다 크고 board 내 corner의 개수가 4보다 클 때 data structure를 업데이트
      if ((residual > boards_3d_[i]->square_size_ * 0.1) &
          (charuco_corners[i].size() > 4)) {
        int board_idx = i;
        // data structure 업데이트
        insertNewBoard(cam_idx, frame_idx, board_idx,
                       charuco_corners[board_idx], charuco_idx[board_idx],
                       frame_path);
      }
    }
  }
}

/**
 * @brief Save all cameras parameters
 *
 * The exports include camera intrinsic matrix, distortion vector,
 * img resolution, pose matrix with respect to the reference camera
 *
 */
void Calibration::saveCamerasParams() {

  std::string save_path_camera_params =
      (!camera_params_file_name_.empty())
          ? save_path_ + camera_params_file_name_
          : save_path_ + "calibrated_cameras_data.yml";
  cv::FileStorage fs(save_path_camera_params, cv::FileStorage::WRITE);
  
  // camera group별로 반복문
  for (const auto &it_cam_group : cam_group_) {
    std::shared_ptr<CameraGroup> cur_cam_group = it_cam_group.second;
    // camera의 수 저장
    fs << "nb_camera" << nb_camera_;
    // 그룹 내 카메라에 대해 반복문
    for (const auto &it_cam : cur_cam_group->cameras_) {
      std::shared_ptr<Camera> cur_cam = it_cam.second.lock();
      // 카메라의 index, intrinsic matrix, distortion 등등 저장
      if (cur_cam) {
        fs << "camera_" + std::to_string(cur_cam->cam_idx_);
        cv::Mat cam_matrix;
        cv::Mat distortion_vector;
        cur_cam->getIntrinsics(cam_matrix, distortion_vector);
        fs << "{"
           << "camera_matrix" << cam_matrix;
        fs << "distortion_vector" << distortion_vector;
        fs << "distortion_type" << cur_cam->distortion_model_;
        fs << "camera_group" << it_cam_group.first;
        fs << "img_width" << cur_cam->im_cols_;
        fs << "img_height" << cur_cam->im_rows_;
        fs << "camera_pose_matrix"
           << cur_cam_group->getCameraPoseMat(cur_cam->cam_idx_).inv() << "}";
      }
    }

    fs.release();
  }
}

/**
 * @brief Save all the 3D object
 *
 * Export 3D points constituting the objects.
 *
 */
void Calibration::save3DObj() {
  std::string save_path_object = save_path_ + "calibrated_objects_data.yml";
  cv::FileStorage fs(save_path_object, cv::FileStorage::WRITE);
  // 각 object에 대해 반복문
  for (const auto &it_obj : object_3d_) {
    std::shared_ptr<Object3D> cur_object = it_obj.second;
    // object id 저장
    fs << "object_" + std::to_string(cur_object->obj_id_);
    int obj_nb_pts = cur_object->nb_pts_;
    cv::Mat pts_mat(3, obj_nb_pts, CV_32FC1);
    //object의 point의 수만큼 반복문 돌면 (x,y,z) 저장
    for (int i = 0; i < obj_nb_pts; i++) {
      cv::Point3f curr_pts = cur_object->pts_3d_[i];
      pts_mat.at<float>(0, i) = curr_pts.x;
      pts_mat.at<float>(1, i) = curr_pts.y;
      pts_mat.at<float>(2, i) = curr_pts.z;
    }
    fs << "{"
       << "points" << pts_mat;
    fs << "}";
  }
  fs.release();
}

/**
 * @brief Save 3D object poses for each frame where the object is visible
 *
 */
void Calibration::save3DObjPose() {
  std::string save_path_object_pose =
      save_path_ + "calibrated_objects_pose_data.yml";
  cv::FileStorage fs(save_path_object_pose, cv::FileStorage::WRITE);
  for (const auto &it_obj : object_3d_) {
    std::shared_ptr<Object3D> cur_object = it_obj.second;
    fs << "object_" + std::to_string(cur_object->obj_id_);
    fs << "{";
    cv::Mat pose_mat(6, cur_object->object_observations_.size(), CV_64FC1);
    int a = 0;
    // 각 object에 대한 rotation, translation parameter 저장
    for (const auto &it_obj_obs : cur_object->object_observations_) {
      std::shared_ptr<Object3DObs> cur_object_obs = it_obj_obs.second.lock();
      if (cur_object_obs) {
        cv::Mat rot, trans;
        cur_object_obs->getPoseVec(rot, trans);
        pose_mat.at<double>(0, a) = rot.at<double>(0);
        pose_mat.at<double>(1, a) = rot.at<double>(1);
        pose_mat.at<double>(2, a) = rot.at<double>(2);
        pose_mat.at<double>(3, a) = trans.at<double>(0);
        pose_mat.at<double>(4, a) = trans.at<double>(1);
        pose_mat.at<double>(5, a) = trans.at<double>(2);
        a = a + 1;
      }
    }
    fs << "poses" << pose_mat;
    fs << "}";
  }
  fs.release();
}

/**
 * @brief Display the board of cam "cam_idx" at frame "frame_idx"
 *
 * @param image image on which to display the detection
 * @param cam_idx camera index
 * @param frame_idx frame index
 */
void Calibration::displayBoards(const cv::Mat image, const int cam_idx,
                                const int frame_idx) {
  //(카메라 index, frame index)의 pair를 cam_frame으로 정의
  std::pair<int, int> cam_frame = std::make_pair(cam_idx, frame_idx);
  std::map<std::pair<int, int>, std::shared_ptr<CameraObs>>::iterator it =
      cams_obs_.find(cam_frame); // Check if a frame exist
  if (it != cams_obs_.end()) {
    // cams_obs_: calibration될 카메라로 key에 (카메라 index, frame index)
    // bboard_observations_: board의 2d points
    for (const auto &it : cams_obs_[cam_frame]->board_observations_) {
      auto board_obs_ptr = it.second.lock();
      if (board_obs_ptr) {
        const std::vector<cv::Point2f> &current_pts = board_obs_ptr->pts_2d_;
        // board observation에 대응되는 3d points
        std::shared_ptr<Board> board_3d_ptr = board_obs_ptr->board_3d_.lock();
        if (board_3d_ptr) {
          std::array<int, 3> &color_temp = board_3d_ptr->color_; //display할 색상
          for (const auto &current_pt : current_pts) {
            LOG_DEBUG << "Pts x :: " << current_pt.x
                      << "   y :: " << current_pt.y;
            // (x,y)에 대해 점 찍기
            cv::circle(image, cv::Point(current_pt.x, current_pt.y), 4,
                       cv::Scalar(color_temp[0], color_temp[1], color_temp[2]),
                       cv::FILLED, 8, 0);
          }
        }
      }
    }
  }
  // cv::imshow("detected board", image);
  // cv::waitKey(1);
}

/**
 * @brief Update the data structure with a new board to be inserted
 *
 * @param cam_idx camera index in which the board was detected
 * @param frame_idx frame index in which the board was detected
 * @param board_idx board index of the detected board
 * @param pts_2d detected 2D points
 * @param charuco_idx index of the detected points in the board
 */
void Calibration::insertNewBoard(const int cam_idx, const int frame_idx,
                                 const int board_idx,
                                 const std::vector<cv::Point2f> pts_2d,
                                 const std::vector<int> charuco_idx,
                                 const std::string frame_path) {
  std::shared_ptr<BoardObs> new_board = std::make_shared<BoardObs>(
      cam_idx, frame_idx, board_idx, pts_2d, charuco_idx, cams_[cam_idx],
      boards_3d_[board_idx]);

  // board 리스트에 추가
  board_observations_[board_observations_.size()] = new_board;

  // 카메라에 board 추가
  cams_[cam_idx]->insertNewBoard(new_board);

  // board 3D에 추가
  boards_3d_[board_idx]->insertNewBoard(new_board);

  // frame 리스트에 추가
  std::map<int, std::shared_ptr<Frame>>::iterator it = frames_.find(
      frame_idx); // 해당 key에 대한 frame이 이미 initialize 되었는지 확인
  if (it != frames_.end()) {
    frames_[frame_idx]->insertNewBoard(
        new_board); // key가 이미 존재하면 push
    frames_[frame_idx]->frame_path_[cam_idx] = frame_path;
  } else {
    std::shared_ptr<Frame> newFrame =
        std::make_shared<Frame>(frame_idx, cam_idx, frame_path);
    frames_[frame_idx] = newFrame; // key가 없으면 새로 만들어준다
    frames_[frame_idx]->insertNewBoard(new_board);
    cams_[cam_idx]->insertNewFrame(newFrame);
    boards_3d_[board_idx]->insertNewFrame(newFrame);
  }

  // cam_obs에 추가. (카메라 index, frame index)로 구성된 cam_fram_idx가 key
  std::pair<int, int> cam_frame_idx = std::make_pair(cam_idx, frame_idx);
  std::map<std::pair<int, int>, std::shared_ptr<CameraObs>>::iterator itCamObs =
      cams_obs_.find(cam_frame_idx); // key가 이미 있는지 확인
  // 이미 존재하면 push
  if (itCamObs != cams_obs_.end()) {
    cams_obs_[cam_frame_idx]->insertNewBoard(new_board);
  // 없으면 만들어준다
  } else {
    std::shared_ptr<CameraObs> new_cam_obs =
        std::make_shared<CameraObs>(new_board);
    cams_obs_[cam_frame_idx] = new_cam_obs;
    frames_[frame_idx]->insertNewCamObs(new_cam_obs);
  }
}

/**
 * @brief Insert a new 3D object observation in the data structure
 *
 * @param new_obj_obs pointer to the new object to be inserted
 */
// 새로운 3d object 추가
void Calibration::insertNewObjectObservation(
    std::shared_ptr<Object3DObs> new_obj_obs) {
  object_observations_[object_observations_.size()] = new_obj_obs;
}

/**
 * @brief Initialize the calibration of all the cameras individually
 *
 * If cam_params_path_ is set and the file exists the initialization is
 * done using the precalibrated information. Otherwise it is done
 * with a subset of images.
 *
 */
void Calibration::initializeCalibrationAllCam() {
  // 파일이 비어있는지, 열 수 있는지 등 확인
  if (!cam_params_path_.empty() && cam_params_path_ != "None") {
    cv::FileStorage fs;
    const bool is_file_available =
        boost::filesystem::exists(cam_params_path_) &&
        cam_params_path_.length() > 0;
    if (!is_file_available) {
      LOG_FATAL << "Camera parameters path '" << cam_params_path_
                << "' doesn't exist.";
      return;
    }
    //precalibrated parameter 담긴 파일 열기
    fs.open(cam_params_path_, cv::FileStorage::READ);

    LOG_INFO << "Initializing camera calibration from " << cam_params_path_;

    for (const auto &it : cams_) {
      // extract camera matrix and distortion coefficients from the file
      // intrinsic 행렬과 distortion 계수 불러오기
      cv::FileNode loaded_cam_params = fs["camera_" + std::to_string(it.first)];

      cv::Mat camera_matrix;
      cv::Mat distortion_coeffs;
      loaded_cam_params["camera_matrix"] >> camera_matrix;
      loaded_cam_params["distortion_vector"] >> distortion_coeffs;

      // 불러온 값들로 초기 setting
      it.second->setIntrinsics(camera_matrix, distortion_coeffs);
    }

  } else {
    LOG_INFO << "Initializing camera calibration using images";

    for (const auto &it : cams_)
      // check!(Intrinsic initilization 하는 부분)
      it.second->initializeCalibration(); 
  }
}

/**
 * @brief Estimate the boards' pose w.r.t. cameras
 *
 * It is based on a PnP algorithm.
 *
 */

// check(ransac으로 pose 추정)
void Calibration::estimatePoseAllBoards() {
  for (const auto &it : board_observations_)
    it.second->estimatePose(ransac_thresh_, nb_iterations_);
}

/**
 * @brief Non-linear refinement of all the cameras intrinsic parameters
 * individually
 *
 */

// check(intrinsic을 refinement)
void Calibration::refineIntrinsicAndPoseAllCam() {
  for (const auto &it : cams_)
    it.second->refineIntrinsicCalibration(nb_iterations_);
}

/**
 * @brief Compute the reprojection error for each boards
 *
 */

// check(reprojection error 계산)
void Calibration::computeReproErrAllBoard() {
  std::vector<float> err_vec;
  float sum_err = 0;
  for (const auto &it : board_observations_) {
    float err = it.second->computeReprojectionError();
  }
}

/**
 * @brief Find all views where multiple boards are visible and store their
 * relative transformation
 *
 * If two boards appears in a single image their interpose can be computed and
 * stored.
 */
// 하나의 카메라에서 관측되는 모든 카메라 간의 pose를 추정
void Calibration::computeBoardsPairPose() {
  board_pose_pairs_.clear();
  // cams_obs_: calibration 될 카메라. key=(camera idx/frame idx)
  for (const auto &it : cams_obs_) {
    std::shared_ptr<CameraObs> current_board = it.second;
    const std::vector<int> &BoardIdx = current_board->board_idx_; //해당 frame, camera로 보이는 3D board의 인덱스

    if (BoardIdx.size() > 1) // board가 여러개 보이는 경우
    {
      // 관측되는 board들 2중 for문 돌면서 id 부여.
      for (const auto &it1 : current_board->board_observations_) {
        auto board1_obs_ptr = it1.second.lock();
        if (board1_obs_ptr) {
          int boardid1 = board1_obs_ptr->board_id_; 
          for (const auto &it2 : current_board->board_observations_) {
            auto board2_obs_ptr = it2.second.lock();
            if (board2_obs_ptr) {
              int boardid2 = board2_obs_ptr->board_id_;

              // 각 board의 카메라와의 pose를 proj_1, proj_2로 정의
              cv::Mat proj_1 = board1_obs_ptr->getPoseMat();
              // 자기 자신과의 관계는 무시
              if (boardid1 != boardid2) // We do not care about the
                                        // transformation with itself ...
              {
                cv::Mat proj_2 = board2_obs_ptr->getPoseMat();
                cv::Mat inter_board_pose = proj_2.inv() * proj_1; // 순환 구조 활용해서 board 간의 pose 계산
                // cam_idx_pair는 (board1의 id, board2의 id)
                std::pair<int, int> cam_idx_pair =
                    std::make_pair(boardid1, boardid2);
                // board_pose_pair에 push. key = (boardind1,boardind2), value = (boardind1,boardind2)
                board_pose_pairs_[cam_idx_pair].push_back(inter_board_pose);
                LOG_DEBUG << "Multiple boards detected";
              }
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Compute the mean transformation between pose pairs
 *
 * Multiple poses can be computed per frames, these measurements are then
 * averaged.
 *
 * @param pose_pairs from {board_pose_pairs_, camera_pose_pairs_,
 * object_pose_pairs_}
 * @param inter_transform from {inter_board_transform_, inter_camera_transform_,
 * inter_object_transform_}
 */

// pose는 각 프레임에서 계산되어 중위값으로 결정된다(논문3)
// pose pairs에는 board-board, camera-camera, object-object에 대한 pairs가 inter_transform에는 pose가 담겨있다.
void Calibration::initInterTransform(
    const std::map<std::pair<int, int>, std::vector<cv::Mat>> &pose_pairs,
    std::map<std::pair<int, int>, cv::Mat> &inter_transform) {
  inter_transform.clear();

  for (const auto &it : pose_pairs) {
    const std::pair<int, int> &pair_idx = it.first;
    const std::vector<cv::Mat> &poses_temp = it.second;
    cv::Mat average_rotation = cv::Mat::zeros(3, 1, CV_64F); //rotation을 matrix 공간
    cv::Mat average_translation = cv::Mat::zeros(3, 1, CV_64F); //translation을 위한 matirx 공간

    // Median
    const size_t num_poses = poses_temp.size();
    std::vector<double> r1, r2, r3;
    std::vector<double> t1, t2, t3;
    // 각 element에 pose의 개수만큼 공간 확보
    r1.reserve(num_poses);
    r2.reserve(num_poses);
    r3.reserve(num_poses);
    t1.reserve(num_poses);
    t2.reserve(num_poses);
    t3.reserve(num_poses);
    
    // 모든 pose들의 R,T에 대한 요소들을 push 
    for (const auto &pose_temp : poses_temp) {
      cv::Mat R, T;
      Proj2RT(pose_temp, R, T);
      r1.push_back(R.at<double>(0));
      r2.push_back(R.at<double>(1));
      r3.push_back(R.at<double>(2));
      t1.push_back(T.at<double>(0));
      t2.push_back(T.at<double>(1));
      t3.push_back(T.at<double>(2));
    }
    // 중위값으로 계산
    average_rotation.at<double>(0) = median(r1);
    average_rotation.at<double>(1) = median(r2);
    average_rotation.at<double>(2) = median(r3);
    average_translation.at<double>(0) = median(t1);
    average_translation.at<double>(1) = median(t2);
    average_translation.at<double>(2) = median(t3);

    //
    inter_transform[pair_idx] =
        RVecT2Proj(average_rotation, average_translation);
    LOG_DEBUG << "Average Rot :: " << average_rotation
              << "    Average Trans :: " << average_translation;
  }
}

/**
 * @brief Initialize the graph with the poses between boards
 *
 */
// inter-board poses를 3D object로 구성하기 위해 directed weighted graph에 저장한다.(논문4)
void Calibration::initInterBoardsGraph() {

  // covis_boards_garph: board간 관계에 대한 그래프. vertex: boardId, edge: number of co-visibility
  covis_boards_graph_.clearGraph();

  // Each board is a vertex if it has been observed at least once
  // 한 벝이라도 관측되는 board는 vertex로 추가
  for (const auto &it : boards_3d_) {
    if (it.second->board_observations_.size() > 0) {
      covis_boards_graph_.addVertex(it.second->board_id_);
    }
  }

  for (const auto &it : board_pose_pairs_) {
    const std::pair<int, int> &board_pair_idx = it.first;
    const std::vector<cv::Mat> &board_poses_temp = it.second;
    // addEdge(v1, v2, weight)로 두 board가 함께 보이는 빈도의 역수를 weight로 저장한다.(다익스트라 알고리즘 통해 shortest path를 찾아 그룹화해주기 때문에 역수로 가중치)
    covis_boards_graph_.addEdge(board_pair_idx.first, board_pair_idx.second,
                                ((double)1 / board_poses_temp.size()));
  }
}

/**
 * @brief Initialize 3D objects using mean inter board pose estimation
 *
 * The connected components in the inter-board graph forms an object. An object
 * is formed of multiple board which are not necessarily physically connected
 * together
 */
void Calibration::init3DObjects() {
  // Find the connected components in the graph (each connected components is a
  // new 3D object)
  std::vector<std::vector<int>> connect_comp =
      covis_boards_graph_.connectedComponents();
  LOG_DEBUG << "Number of 3D objects detected :: " << connect_comp.size();

  // Declare a new 3D object for each connected component
  for (int i = 0; i < connect_comp.size(); i++) {
    LOG_DEBUG << "Obj Id :: " << i;
    LOG_DEBUG << "Number of boards in this object :: "
              << connect_comp[i].size();

    // Find the reference board in this object
    int ref_board_id =
        *min_element(connect_comp[i].begin(), connect_comp[i].end());

    // Declare a new 3D object
    std::shared_ptr<Object3D> newObject3D =
        std::make_shared<Object3D>(connect_comp[i].size(), ref_board_id, i,
                                   boards_3d_[ref_board_id]->color_);
    int pts_count = 0;

    // Compute the shortest path between the reference and the other board
    for (int j = 0; j < connect_comp[i].size(); j++) {
      int current_board_id = connect_comp[i][j];
      newObject3D->insertBoardInObject(boards_3d_[current_board_id]);

      // Compute the transformation between the reference board and the other
      // boards in the object if the board is not the referential board compute
      // the path
      std::vector<int> short_path = covis_boards_graph_.shortestPathBetween(
          ref_board_id, current_board_id);
      // Compute the transformation wrt. the reference board
      cv::Mat transform = (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0,
                           0, 1, 0, 0, 0, 0,
                           1); // initialize the transformation to identity
      for (int k = 0; k < short_path.size() - 1; k++) {
        int current_board = short_path[k];
        int next_board = short_path[k + 1];
        std::pair<int, int> board_pair_idx =
            std::make_pair(current_board, next_board);
        cv::Mat current_trans = inter_board_transform_[board_pair_idx];
        transform = transform * current_trans.inv();
      }

      // Store the relative board transformation in the object
      newObject3D->setBoardPoseMat(transform, current_board_id);

      // Transform the 3D pts to push in the object 3D
      std::vector<cv::Point3f> trans_pts =
          transform3DPts(boards_3d_[current_board_id]->pts_3d_,
                         newObject3D->getBoardRotVec(current_board_id),
                         newObject3D->getBoardTransVec(current_board_id));
      // Make a indexing between board to object
      for (int k = 0; k < trans_pts.size(); k++) {
        int char_id = k;
        std::pair<int, int> boardid_charid =
            std::make_pair(current_board_id, char_id);
        newObject3D->pts_board_2_obj_[boardid_charid] = pts_count;
        newObject3D->pts_obj_2_board_.push_back(boardid_charid);
        newObject3D->pts_3d_.push_back(trans_pts[k]);
        pts_count++;
      }
      LOG_DEBUG << "Board ID :: " << current_board_id;
    }
    // Add the 3D object into the structure
    object_3d_[i] = newObject3D;
  }
}

/**
 * @brief Initialize 3D objects observation
 *
 * After the "physical" 3D object have been initialized, we gather all the
 * boards observation (belonging to the object) into the 3D object observation.
 *
 * @param object_idx index of the 3D object observed
 */
void Calibration::init3DObjectObs(const int object_idx) {

  // Iterate through cameraobs
  for (const auto &it_cam_obs : cams_obs_) {
    std::pair<int, int> cam_id_frame_id = it_cam_obs.first; // Cam ind/Frame ind
    std::shared_ptr<CameraObs> current_camobs = it_cam_obs.second;

    // Declare the 3D object observed in this camera observation
    // Keep in mind that a single object can be observed in one image
    std::shared_ptr<Object3DObs> object_obs =
        std::make_shared<Object3DObs>(object_3d_[object_idx], object_idx);

    // Check the boards observing this camera
    std::map<int, std::weak_ptr<BoardObs>> current_board_obs =
        current_camobs->board_observations_;
    for (const auto &it_board_obs : current_board_obs) {
      auto board_obs_ptr = it_board_obs.second.lock();
      if (board_obs_ptr) {
        // Check if this board correspond to the object of interest
        std::map<int, std::weak_ptr<Board>>::iterator it =
            object_3d_[object_idx]->boards_.find(board_obs_ptr->board_id_);
        if (it != object_3d_[object_idx]
                      ->boards_.end()) // if the board belong to the object
        {
          object_obs->insertNewBoardObs(board_obs_ptr);
        }
      }
    }

    if (object_obs->pts_id_.size() > 0) {
      // Update the camobs//frame//camera//3DObject
      cams_obs_[cam_id_frame_id]->insertNewObject(object_obs);
      frames_[cam_id_frame_id.second]->insertNewObject(object_obs);
      cams_[it_cam_obs.first.first]->insertNewObject(object_obs);
      object_3d_[object_idx]->insertNewObject(object_obs);
      object_3d_[object_idx]->insertNewFrame(frames_[cam_id_frame_id.second]);
      insertNewObjectObservation(object_obs);
    }
  }
}

/**
 * @brief Initialize all 3D object observations
 *
 */
void Calibration::initAll3DObjectObs() {
  for (const auto &it : object_3d_)
    this->init3DObjectObs(it.first);
}

/**
 * @brief Estimate all the pose of 3D object observation using a PnP algo
 *
 */
void Calibration::estimatePoseAllObjects() {
  for (const auto &it : object_observations_)
    it.second->estimatePose(ransac_thresh_, nb_iterations_);
}

/**
 * @brief Compute the reprojection error for each object
 *
 */
void Calibration::computeReproErrAllObject() {
  std::vector<float> err_vec;
  err_vec.reserve(object_observations_.size());
  for (const auto &it : object_observations_)
    err_vec.push_back(it.second->computeReprojectionError());

  LOG_INFO << "Mean Error "
           << std::accumulate(err_vec.begin(), err_vec.end(), 0.0) /
                  err_vec.size();
}

/**
 * @brief Refine the structure of all the 3D objects
 *
 */
void Calibration::refineAllObject3D() {
  for (const auto &it : object_3d_)
    it.second->refineObject(nb_iterations_);
}

/**
 * @brief Find all view where multiple camera share visible objects and store
 * their relative transformation
 *
 */
void Calibration::computeCamerasPairPose() {
  camera_pose_pairs_.clear();
  // Iterate through frames
  for (const auto &it_frame : frames_) {
    // if more than one observation is available
    if (it_frame.second->board_observations_.size() > 1) {
      // Iterate through the object observation
      std::map<int, std::weak_ptr<Object3DObs>> frame_obj_obs =
          it_frame.second->object_observations_;
      for (const auto &it_objectobs1 : frame_obj_obs) {
        auto obj_obs_1_ptr = it_objectobs1.second.lock();
        if (obj_obs_1_ptr) {
          int cam_id_1 = obj_obs_1_ptr->camera_id_;
          int obj_id_1 = obj_obs_1_ptr->object_3d_id_;
          cv::Mat pose_cam_1 = obj_obs_1_ptr->getPoseMat();
          for (const auto &it_objectobs2 : frame_obj_obs) {
            auto obj_obs_2_ptr = it_objectobs2.second.lock();
            if (obj_obs_2_ptr) {
              int cam_id_2 = obj_obs_2_ptr->camera_id_;
              int obj_id_2 = obj_obs_2_ptr->object_3d_id_;
              cv::Mat pose_cam_2 = obj_obs_2_ptr->getPoseMat();
              if (cam_id_1 != cam_id_2) // if the camera is not the same
              {
                // if the same object is visible from the two cameras
                if (obj_id_1 == obj_id_2) {
                  // Compute the relative pose between the cameras
                  cv::Mat inter_cam_pose =
                      pose_cam_2 * pose_cam_1.inv(); // not sure here ...

                  // Store in a database
                  camera_pose_pairs_[std::make_pair(cam_id_1, cam_id_2)]
                      .push_back(inter_cam_pose);
                }
              }
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Initialize the relationship graph between cameras to form groups
 *
 */
void Calibration::initInterCamerasGraph() {
  covis_camera_graph_.clearGraph();
  // Each camera is a vertex if it has observed at least one object
  for (const auto &it : this->cams_) {
    if (it.second->board_observations_.size() > 0) {
      covis_camera_graph_.addVertex(it.second->cam_idx_);
    }
  }
  // Build the graph with cameras' pairs
  for (const auto &it : camera_pose_pairs_) {
    const std::pair<int, int> &camera_pair_idx = it.first;
    const std::vector<cv::Mat> &camera_poses_temp = it.second;
    covis_camera_graph_.addEdge(camera_pair_idx.first, camera_pair_idx.second,
                                ((double)1 / camera_poses_temp.size()));
  }
}

/**
 * @brief Initialize camera group based on co-visibility pair between cameras
 *
 */
void Calibration::initCameraGroup() {
  // Find the connected components in the graph (each connected components is a
  // new camera group)
  std::vector<std::vector<int>> connect_comp =
      covis_camera_graph_.connectedComponents();
  LOG_DEBUG << "Number of camera group detected :: " << connect_comp.size();

  // Declare a new camera group for each connected component
  for (int i = 0; i < connect_comp.size(); i++) {
    LOG_DEBUG << "camera group id :: " << i;
    LOG_DEBUG << "Number of cameras in the group :: " << connect_comp.size();

    // Find the reference camera in this group
    int id_ref_cam =
        *min_element(connect_comp[i].begin(), connect_comp[i].end());

    // Declare a new camera group
    std::shared_ptr<CameraGroup> new_camera_group =
        std::make_shared<CameraGroup>(id_ref_cam, i);

    // Compute the shortest path between the reference and the other cams
    for (int j = 0; j < connect_comp[i].size(); j++) {
      int current_camera_id = connect_comp[i][j];
      new_camera_group->insertCamera(cams_[current_camera_id]);

      // Compute the transformation between the reference cam and the other
      // cam in the group
      std::vector<int> short_path = covis_camera_graph_.shortestPathBetween(
          id_ref_cam, current_camera_id);
      // Compute the transformation wrt. the reference camera
      cv::Mat transform =
          (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 1); // initialize the transformation to identity
      for (int k = 0; k < short_path.size() - 1; k++) {
        int current_cam = short_path[k];
        int next_cam = short_path[k + 1];
        std::pair<int, int> cam_pair_idx =
            std::make_pair(current_cam, next_cam);
        cv::Mat current_trans = inter_camera_transform_[cam_pair_idx];
        // transform = transform * current_trans.inv();
        transform = transform * current_trans;
      }
      // Store the relative camera transformation in the object
      new_camera_group->setCameraPoseMat(transform, current_camera_id);
    }
    // Add the 3D camera group into the structure
    cam_group_[i] = new_camera_group;
  }
}

/**
 * @brief Initialize camera group observation
 *
 * @param camera_group_idx camera group index
 */
void Calibration::initCameraGroupObs(const int camera_group_idx) {
  // List of camera idx in the group
  const std::vector<int> &cam_in_group = cam_group_[camera_group_idx]->cam_idx;

  // Iterate through frame
  for (const auto &it_frame : frames_) {
    int current_frame_id = it_frame.second->frame_idx_;
    std::shared_ptr<CameraGroupObs> new_cam_group_obs =
        std::make_shared<CameraGroupObs>(
            cam_group_[camera_group_idx]); // declare a new observation

    std::map<int, std::weak_ptr<Object3DObs>> current_object_obs =
        it_frame.second->object_observations_;
    for (const auto &it_obj_obs : current_object_obs) {
      auto obj_obs_ptr = it_obj_obs.second.lock();
      if (obj_obs_ptr) {
        int current_cam_id = obj_obs_ptr->camera_id_;
        int current_obj_id = obj_obs_ptr->object_3d_id_;

        // Check if this camera id belongs to the group
        if (std::find(cam_in_group.begin(), cam_in_group.end(),
                      current_cam_id) != cam_in_group.end()) {
          // if (count(cam_in_group.begin(), cam_in_group.end(),
          // current_cam_id))
          // {
          // the camera is in the group so this object is visible in the cam
          // group udpate the observation
          new_cam_group_obs->insertObjectObservation(obj_obs_ptr);

          // push the object observation in the camera group
          cam_group_[camera_group_idx]->insertNewObjectObservation(obj_obs_ptr);
        }
      }
    }
    if (new_cam_group_obs->object_observations_.size() > 0) {
      // add the new cam group observation to the database
      cams_group_obs_[std::make_pair(camera_group_idx, current_frame_id)] =
          new_cam_group_obs;

      // update frame
      frames_[current_frame_id]->insertNewCameraGroupObs(new_cam_group_obs,
                                                         camera_group_idx);

      // update the cam_group
      cam_group_[camera_group_idx]->insertNewFrame(frames_[current_frame_id]);
    }
  }
}

/**
 * @brief Initialize all the camera groups observations
 *
 */
void Calibration::initAllCameraGroupObs() {
  for (const auto &it : cam_group_) {
    int camera_group_idx = it.second->cam_group_idx_;
    initCameraGroupObs(camera_group_idx);
  }
}

/**
 * @brief Non-linear optimization of the camera pose in the groups and the pose
 * of the observed objects
 *
 */
void Calibration::refineAllCameraGroup() {
  for (const auto &it : cam_group_) {
    // it->second->computeObjPoseInCameraGroup();
    it.second->refineCameraGroup(nb_iterations_);
  }

  // Update the object3D observation
  for (const auto &it : cams_group_obs_)
    it.second->updateObjObsPose();
}

/**
 * @brief Find the pair of objects to use to calibrate the pairs of camera
 * groups
 *
 * The pair of object with the highest number of occurrences are used for
 * calibration.
 */
void Calibration::findPairObjectForNonOverlap() {
  no_overlap_object_pair_.clear();
  // Iterate through the camera groups
  for (const auto &it_groups_1 : cam_group_) {
    int group_idx1 = it_groups_1.first;
    for (const auto &it_groups_2 : cam_group_) {
      int group_idx2 = it_groups_2.first;
      if (group_idx1 != group_idx2) // if the two groups are different
      {
        // Prepare the list of possible objects pairs
        std::map<std::pair<int, int>, int> count_pair_obs;
        for (const auto &it_obj1 : object_3d_) {
          for (const auto &it_obj2 : object_3d_) {
            int obj_id1 = it_obj1.second->obj_id_;
            int obj_id2 = it_obj2.second->obj_id_;
            if (obj_id1 != obj_id2)
              count_pair_obs[std::make_pair(obj_id1, obj_id2)] = 0;
          }
        }

        // move to shared_ptr cause there is no = for weak_ptr
        std::map<int, std::shared_ptr<Frame>> it_groups_1_frames;
        std::map<int, std::shared_ptr<Frame>> it_groups_2_frames;
        for (const auto &item : it_groups_1.second->frames_) {
          if (auto frame_ptr = item.second.lock())
            it_groups_1_frames[item.first] = frame_ptr;
        }
        for (const auto &item : it_groups_2.second->frames_) {
          if (auto frame_ptr = item.second.lock())
            it_groups_2_frames[item.first] = frame_ptr;
        }

        // Find frames in common
        std::map<int, std::shared_ptr<Frame>> common_frames;
        std::map<int, std::shared_ptr<Frame>>::iterator it_frames(
            common_frames.begin());
        std::set_intersection(
            it_groups_1_frames.begin(), it_groups_1_frames.end(),
            it_groups_2_frames.begin(), it_groups_2_frames.end(),
            std::inserter(common_frames, it_frames));

        // Iterate through the frames and count the occurrence of object pair
        // (to select the pair of object appearing the most)
        for (const auto &it_common_frames : common_frames) {
          // Find the index of the observation corresponding to the groups in
          // cam group obs
          auto it_camgroupid_1 =
              find(it_common_frames.second->cam_group_idx_.begin(),
                   it_common_frames.second->cam_group_idx_.end(), group_idx1);
          auto it_camgroupid_2 =
              find(it_common_frames.second->cam_group_idx_.begin(),
                   it_common_frames.second->cam_group_idx_.end(), group_idx2);
          int index_camgroup_1 =
              it_camgroupid_1 - it_common_frames.second->cam_group_idx_.begin();
          int index_camgroup_2 =
              it_camgroupid_2 - it_common_frames.second->cam_group_idx_.begin();

          // Access the objects 3D index for both groups
          auto common_frames_cam_group1_obs_ptr =
              it_common_frames.second->cam_group_observations_[index_camgroup_1]
                  .lock();
          auto common_frames_cam_group2_obs_ptr =
              it_common_frames.second->cam_group_observations_[index_camgroup_2]
                  .lock();
          if (common_frames_cam_group1_obs_ptr &&
              common_frames_cam_group2_obs_ptr) {
            std::map<int, std::weak_ptr<Object3DObs>> object_obs_1 =
                common_frames_cam_group1_obs_ptr->object_observations_;
            std::map<int, std::weak_ptr<Object3DObs>> object_obs_2 =
                common_frames_cam_group2_obs_ptr->object_observations_;
            for (const auto &it_object_obs_1 : object_obs_1) {
              auto object_obs_1_ptr = it_object_obs_1.second.lock();
              if (object_obs_1_ptr) {
                int obj_ind_1 = object_obs_1_ptr->object_3d_id_;
                for (const auto &it_object_obs_2 : object_obs_2) {
                  auto object_obs_2_ptr = it_object_obs_2.second.lock();
                  if (object_obs_2_ptr) {
                    int obj_ind_2 = object_obs_2_ptr->object_3d_id_;
                    count_pair_obs[std::make_pair(obj_ind_1, obj_ind_2)]++;
                  }
                }
              }
            }
          }
        }

        // find the pair of object with the maximum shared frames
        unsigned currentMax = 0;
        std::pair<int, int> arg_max = std::make_pair(0, 0);
        for (const auto &it : count_pair_obs) {
          if (it.second > currentMax) {
            arg_max = it.first;
            currentMax = it.second;
          }
        }

        // Save in the data structure
        LOG_DEBUG << "max visibility, Object 1 :: " << arg_max.first
                  << "Object 2 :: " << arg_max.second;
        LOG_DEBUG << "Number of occurrence :: " << currentMax;
        no_overlap_object_pair_[std::make_pair(group_idx1, group_idx2)] =
            arg_max;
      }
    }
  }
}

/**
 * @brief Handeye calibration of a pair of non overlapping pair of group of
 * cameras
 *
 */
void Calibration::initNonOverlapPair(const int cam_group_id1,
                                     const int cam_group_id2) {
  // Prepare the group of interest
  std::shared_ptr<CameraGroup> cam_group1 = cam_group_[cam_group_id1];
  std::shared_ptr<CameraGroup> cam_group2 = cam_group_[cam_group_id2];

  // Check the object per camera
  std::pair<int, int> object_pair =
      no_overlap_object_pair_[std::make_pair(cam_group_id1, cam_group_id2)];
  int object_cam_1 = object_pair.first;
  int object_cam_2 = object_pair.second;

  // Prepare the 3D objects
  std::shared_ptr<Object3D> object_3D_1 = object_3d_[object_cam_1];
  std::shared_ptr<Object3D> object_3D_2 = object_3d_[object_cam_2];
  const std::vector<cv::Point3f> &pts_3d_obj_1 = object_3D_1->pts_3d_;
  const std::vector<cv::Point3f> &pts_3d_obj_2 = object_3D_2->pts_3d_;

  // std::vector to store data for non-overlapping calibration
  std::vector<cv::Mat> pose_abs_1,
      pose_abs_2; // absolute pose stored to compute relative displacements
  cv::Mat repo_obj_1_2; // reprojected pts for clustering

  // move to shared_ptr cause there is no = for weak_ptr
  std::map<int, std::shared_ptr<Frame>> cam_group1_frames;
  std::map<int, std::shared_ptr<Frame>> cam_group2_frames;
  for (const auto &item : cam_group1->frames_) {
    if (auto frames_ptr = item.second.lock())
      cam_group1_frames[item.first] = frames_ptr;
  }
  for (const auto &item : cam_group2->frames_) {
    if (auto frames_ptr = item.second.lock())
      cam_group2_frames[item.first] = frames_ptr;
  }

  // Find frames in common
  std::map<int, std::shared_ptr<Frame>> common_frames;
  std::map<int, std::shared_ptr<Frame>>::iterator it_frames(
      common_frames.begin());
  std::set_intersection(cam_group1_frames.begin(), cam_group1_frames.end(),
                        cam_group2_frames.begin(), cam_group2_frames.end(),
                        std::inserter(common_frames, it_frames));

  // Iterate through common frames and reproject the objects in the images to
  // cluster
  for (const auto &it_common_frames : common_frames) {
    // Find the index of the observation corresponding to the groups in cam
    // group obs
    auto it_camgroupid_1 =
        find(it_common_frames.second->cam_group_idx_.begin(),
             it_common_frames.second->cam_group_idx_.end(), cam_group_id1);
    auto it_camgroupid_2 =
        find(it_common_frames.second->cam_group_idx_.begin(),
             it_common_frames.second->cam_group_idx_.end(), cam_group_id2);
    int index_camgroup_1 =
        it_camgroupid_1 - it_common_frames.second->cam_group_idx_.begin();
    int index_camgroup_2 =
        it_camgroupid_2 - it_common_frames.second->cam_group_idx_.begin();

    // check if both objects of interest are in the frame
    std::weak_ptr<CameraGroupObs> cam_group_obs1 =
        it_common_frames.second->cam_group_observations_[index_camgroup_1];
    std::weak_ptr<CameraGroupObs> cam_group_obs2 =
        it_common_frames.second->cam_group_observations_[index_camgroup_2];

    auto cam_group_obs1_ptr = cam_group_obs1.lock();
    auto cam_group_obs2_ptr = cam_group_obs2.lock();
    if (cam_group_obs1_ptr && cam_group_obs2_ptr) {
      const std::vector<int> &cam_group_obs_obj1 =
          cam_group_obs1_ptr->object_idx_;
      const std::vector<int> &cam_group_obs_obj2 =
          cam_group_obs2_ptr->object_idx_;
      auto it1 = find(cam_group_obs_obj1.begin(), cam_group_obs_obj1.end(),
                      object_cam_1);
      auto it2 = find(cam_group_obs_obj2.begin(), cam_group_obs_obj2.end(),
                      object_cam_2);
      bool obj_vis1 = it1 != cam_group_obs_obj1.end();
      bool obj_vis2 = it2 != cam_group_obs_obj2.end();
      int index_objobs_1 = it1 - cam_group_obs_obj1.begin();
      int index_objobs_2 = it2 - cam_group_obs_obj2.begin();

      // if both objects are visible
      if (obj_vis1 & obj_vis2) {
        // Reproject 3D objects in ref camera group
        auto cam_group_obs1_cam_group_ptr =
            cam_group_obs1_ptr->cam_group_.lock();
        auto cam_group_obs2_cam_group_ptr =
            cam_group_obs2_ptr->cam_group_.lock();
        if (cam_group_obs1_cam_group_ptr && cam_group_obs2_cam_group_ptr) {
          std::weak_ptr<Camera> ref_cam_1 =
              cam_group_obs1_cam_group_ptr
                  ->cameras_[cam_group_obs1_cam_group_ptr->id_ref_cam_];
          std::weak_ptr<Camera> ref_cam_2 =
              cam_group_obs2_cam_group_ptr
                  ->cameras_[cam_group_obs2_cam_group_ptr->id_ref_cam_];

          auto obj_obs1_ptr =
              cam_group_obs1_ptr->object_observations_[index_objobs_1].lock();
          auto obj_obs2_ptr =
              cam_group_obs2_ptr->object_observations_[index_objobs_2].lock();
          if (obj_obs1_ptr && obj_obs2_ptr) {
            int object_id1 = obj_obs1_ptr->object_3d_id_;
            int object_id2 = obj_obs2_ptr->object_3d_id_;
            cv::Mat pose_obj_1 =
                cam_group_obs1_ptr->getObjectPoseMat(object_id1);
            cv::Mat pose_obj_2 =
                cam_group_obs2_ptr->getObjectPoseMat(object_id2);
            pose_abs_1.push_back(pose_obj_1);
            pose_abs_2.push_back(pose_obj_2);
          }
        }
      }
    }
  }

  // HANDEYE CALIBRATION
  cv::Mat pose_g1_g2;
  if (he_approach_ == 0) {
    // Boot strapping technique
    int nb_cluster = 20;
    int nb_it_he = 200; // Nb of time we apply the handeye calibration
    pose_g1_g2 = handeyeBootstratpTranslationCalibration(
        nb_cluster, nb_it_he, pose_abs_1, pose_abs_2);
  } else {
    pose_g1_g2 = handeyeCalibration(pose_abs_1, pose_abs_2);
  }

  pose_g1_g2 = pose_g1_g2.inv();
  // Save the parameter in the datastructure
  no_overlap_camgroup_pair_pose_[std::make_pair(cam_group_id1, cam_group_id2)] =
      pose_g1_g2;
  no_overlap__camgroup_pair_common_cnt_[std::make_pair(
      cam_group_id1, cam_group_id2)] = pose_abs_1.size();
}

/**
 * @brief Initialize the pose between all groups of non-overlaping camera group
 */
void Calibration::findPoseNoOverlapAllCamGroup() {
  no_overlap_camgroup_pair_pose_.clear();
  no_overlap_camgroup_pair_pose_.clear();

  // Iterate through the camera groups
  for (const auto &it_groups_1 : cam_group_) {
    int group_idx1 = it_groups_1.first;
    for (const auto &it_groups_2 : cam_group_) {
      int group_idx2 = it_groups_2.first;
      if (group_idx1 != group_idx2) // if the two groups are different
      {
        initNonOverlapPair(group_idx1, group_idx2);
      }
    }
  }
}

/**
 * @brief Create graph between nonoverlap camera groups
 *
 */
void Calibration::initInterCamGroupGraph() {
  no_overlap_camgroup_graph_.clearGraph();
  // All the existing groups form a vertex
  for (const auto &it : cam_group_) {
    if (it.second->object_observations_.size() > 0) {
      no_overlap_camgroup_graph_.addVertex(it.second->cam_group_idx_);
    }
  }

  // Create the graph
  for (const auto &it : no_overlap_camgroup_pair_pose_) {
    const std::pair<int, int> &camgroup_pair_idx = it.first;
    const cv::Mat &camgroup_poses_temp = it.second;
    int nb_common_frame =
        no_overlap__camgroup_pair_common_cnt_[camgroup_pair_idx];
    no_overlap_camgroup_graph_.addEdge(camgroup_pair_idx.first,
                                       camgroup_pair_idx.second,
                                       ((double)1 / nb_common_frame));
  }
}

/**
 * @brief Merge all camera groups using non-overlaping pose estimation
 *
 */
void Calibration::mergeCameraGroup() {
  // Find the connected components in the graph
  std::vector<std::vector<int>> connect_comp =
      no_overlap_camgroup_graph_.connectedComponents();
  std::map<int, std::shared_ptr<CameraGroup>> cam_group; // list of camera group

  for (int i = 0; i < connect_comp.size(); i++) {
    // find the reference camera group reference and the camera reference among
    // all the groups
    int id_ref_cam_group =
        *std::min_element(connect_comp[i].begin(), connect_comp[i].end());
    int id_ref_cam = cam_group_[id_ref_cam_group]->id_ref_cam_;

    // Recompute the camera pose in the referential of the reference group
    std::map<int, cv::Mat>
        cam_group_pose_to_ref; // pose of the cam group in the cam group

    // Used the graph to find the transformations of camera groups to the
    // reference group
    for (int j = 0; j < connect_comp[i].size(); j++) {
      int current_cam_group_id = connect_comp[i][j];
      // Compute the transformation between the reference group and the current
      // group
      std::vector<int> short_path =
          no_overlap_camgroup_graph_.shortestPathBetween(id_ref_cam_group,
                                                         current_cam_group_id);
      // Compute the transformation wrt. the reference camera
      cv::Mat transform =
          (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 1); // initialize the transformation to identity
      for (int k = 0; k < short_path.size() - 1; k++) {
        int current_group = short_path[k];
        int next_group = short_path[k + 1];
        std::pair<int, int> group_pair_idx =
            std::make_pair(current_group, next_group);
        cv::Mat current_trans = no_overlap_camgroup_pair_pose_[group_pair_idx];
        // transform = transform * current_trans.inv();
        transform = transform * current_trans;
      }
      // Store the poses
      cam_group_pose_to_ref[current_cam_group_id] = transform;
    }

    // initialize the camera group
    std::shared_ptr<CameraGroup> new_camera_group =
        std::make_shared<CameraGroup>(id_ref_cam, i);
    // Iterate through the camera groups and add all the cameras individually in
    // the new group
    for (const auto &it_group : cam_group_) {
      // Check if the current camera group belong to the final group
      int current_cam_group_idx = it_group.second->cam_group_idx_;
      if (std::find(connect_comp[i].begin(), connect_comp[i].end(),
                    current_cam_group_idx) != connect_comp[i].end()) {
        // Prepare the current group pose in the referential of the final group
        std::shared_ptr<CameraGroup> current_group = it_group.second;
        cv::Mat pose_in_final =
            cam_group_pose_to_ref[current_group->cam_group_idx_];
        // the camera group is in the final group so we include its cameras
        for (const auto &it_cam : current_group->cameras_) {
          std::shared_ptr<Camera> current_camera = it_cam.second.lock();
          if (current_camera) {
            // Update the pose in the referential of the final group
            cv::Mat pose_cam_in_current_group =
                current_group->getCameraPoseMat(current_camera->cam_idx_);
            cv::Mat transform = pose_cam_in_current_group * pose_in_final;
            new_camera_group->insertCamera(current_camera);
            new_camera_group->setCameraPoseMat(transform,
                                               current_camera->cam_idx_);
          }
        }
      }
    }
    cam_group[i] = new_camera_group;
  }
  // Erase previous camera group and replace it with the merged one
  cam_group_.clear();
  cam_group_ = cam_group;
}

/**
 * @brief Merge the camera groups observation
 *
 */
void Calibration::mergeAllCameraGroupObs() {
  // First we erase all the Cameragroups observation in the entire datastructure
  for (const auto &it_frame : frames_) {
    it_frame.second->cam_group_idx_.clear();
    it_frame.second->cam_group_observations_.clear();
  }
  cams_group_obs_.clear();

  // Reinitialize all camera obserations
  for (const auto &it : cam_group_) {
    int camera_group_idx = it.second->cam_group_idx_;
    initCameraGroupObs(camera_group_idx);
  }
}

/**
 * @brief Compute the 3D object position in the camera group
 *
 */
void Calibration::computeAllObjPoseInCameraGroup() {
  for (const auto &it : cam_group_)
    it.second->computeObjPoseInCameraGroup();
  // Compute the pose of each object in the camera groups obs
  for (const auto &it_cam_group_obs : cams_group_obs_)
    it_cam_group_obs.second->computeObjectsPose();
}

/**
 * @brief Find all frames where multiple objects are visible and store their
 * relative transformation.
 *
 * If two object appears in a single frames their
 * interpose can be computed and stored.
 */
void Calibration::computeObjectsPairPose() {
  object_pose_pairs_.clear();
  // Iterate through camera group obs
  for (const auto &it_cam_group_obs : cams_group_obs_) {
    if (it_cam_group_obs.second->object_idx_.size() > 1) {
      std::map<int, std::weak_ptr<Object3DObs>> obj_obs =
          it_cam_group_obs.second->object_observations_;
      for (const auto &it_object1 : obj_obs) {
        auto it_object1_ptr = it_object1.second.lock();
        if (it_object1_ptr) {
          int object_3d_id_1 = it_object1_ptr->object_3d_id_;
          cv::Mat obj_pose_1 =
              it_cam_group_obs.second->getObjectPoseMat(object_3d_id_1);
          // cv::Mat obj_pose_1 = it_object1->second->getPoseInGroupMat();
          for (const auto &it_object2 : obj_obs) {
            auto it_object2_ptr = it_object2.second.lock();
            if (it_object2_ptr) {
              int object_3d_id_2 = it_object2_ptr->object_3d_id_;
              cv::Mat obj_pose_2 =
                  it_cam_group_obs.second->getObjectPoseMat(object_3d_id_2);
              // cv::Mat obj_pose_2 = it_object2->second->getPoseInGroupMat();
              if (object_3d_id_1 != object_3d_id_2) {
                cv::Mat inter_object_pose = obj_pose_2.inv() * obj_pose_1;
                std::pair<int, int> object_idx_pair =
                    std::make_pair(object_3d_id_1, object_3d_id_2);
                object_pose_pairs_[object_idx_pair].push_back(
                    inter_object_pose);
              }
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Initialize the graph with the poses between objects
 *
 */
void Calibration::initInterObjectsGraph() {

  covis_objects_graph_.clearGraph();
  // Each object is a vertex if it has been observed at least once
  for (const auto &it : object_3d_) {
    if (it.second->object_observations_.size() > 0) {
      covis_objects_graph_.addVertex(it.second->obj_id_);
    }
  }

  for (const auto &it : object_pose_pairs_) {
    const std::pair<int, int> &object_pair_idx = it.first;
    const std::vector<cv::Mat> &object_poses_temp = it.second;
    covis_objects_graph_.addEdge(object_pair_idx.first, object_pair_idx.second,
                                 ((double)1 / (object_poses_temp.size())));
  }
  LOG_DEBUG << "GRAPH INTER OBJECT DONE";
}

/**
 * @brief Merge all objects groups which have been visible in same camera groups
 *
 */
void Calibration::mergeObjects() {
  // find the connected objects in the graph
  std::vector<std::vector<int>> connect_comp =
      covis_objects_graph_.connectedComponents();
  std::map<int, std::shared_ptr<Object3D>> object_3d; // list of object 3D

  for (int i = 0; i < connect_comp.size(); i++) {
    // find the reference camera group reference and the camera reference among
    // all the groups
    int id_ref_object =
        *std::min_element(connect_comp[i].begin(), connect_comp[i].end());
    int ref_board_id = object_3d_[id_ref_object]->ref_board_id_;

    // recompute the board poses in the referential of the reference object
    std::map<int, cv::Mat>
        object_pose_to_ref; // pose of the object in the ref object
    int nb_board_in_obj = 0;

    // Used the graph to find the transformations of objects to the reference
    // object
    for (int j = 0; j < connect_comp[i].size(); j++) {
      nb_board_in_obj += object_3d_[connect_comp[i][j]]->boards_.size();
      int current_object_id = connect_comp[i][j];
      // Compute the transformation between the reference object and the current
      // object
      std::vector<int> short_path = covis_objects_graph_.shortestPathBetween(
          id_ref_object, current_object_id);
      // Compute the transformation wrt. the reference object
      cv::Mat transform =
          (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 1); // initialize the transformation to identity
      for (int k = 0; k < short_path.size() - 1; k++) {
        int current_object = short_path[k];
        int next_object = short_path[k + 1];
        std::pair<int, int> object_pair_idx =
            std::make_pair(current_object, next_object);
        cv::Mat current_trans = inter_object_transform_[object_pair_idx];
        transform = transform * current_trans.inv(); // original
        // transform = transform * current_trans;
      }
      // Store the poses
      object_pose_to_ref[current_object_id] = transform;
    }
    // initialize the object
    std::shared_ptr<Object3D> newObject3D = std::make_shared<Object3D>(
        nb_board_in_obj, ref_board_id, i, boards_3d_[ref_board_id]->color_);
    int pts_count = 0;
    // Iterate through the objects and add all of them individually in the new
    // object
    for (const auto &it_object : object_3d_) {
      // Check if the current object belong to the new object
      int current_object_idx = it_object.second->obj_id_;
      if (std::find(connect_comp[i].begin(), connect_comp[i].end(),
                    current_object_idx) != connect_comp[i].end()) {
        // Prepare the current object pose in the referential of the merged
        // object
        std::shared_ptr<Object3D> current_object = it_object.second;
        cv::Mat pose_in_merged = object_pose_to_ref[current_object->obj_id_];
        // the object is in the merged group so we include its boards
        for (const auto &it_board : current_object->boards_) {
          std::shared_ptr<Board> current_board = it_board.second.lock();
          if (current_board) {
            // Update the pose to be in the referential of the merged object
            cv::Mat pose_board_in_current_obj =
                current_object->getBoardPoseMat(current_board->board_id_);
            cv::Mat transform = pose_in_merged * pose_board_in_current_obj;

            // insert new board
            newObject3D->insertBoardInObject(current_board);
            // Store the relative board transformation in the object
            newObject3D->setBoardPoseMat(transform, current_board->board_id_);
            // Transform the 3D pts to push in the object 3D
            std::vector<cv::Point3f> trans_pts = transform3DPts(
                current_board->pts_3d_,
                newObject3D->getBoardRotVec(current_board->board_id_),
                newObject3D->getBoardTransVec(current_board->board_id_));
            // Make a indexing between board to object
            for (int k = 0; k < trans_pts.size(); k++) {
              int char_id = k;
              std::pair<int, int> boardid_charid =
                  std::make_pair(current_board->board_id_, char_id);
              newObject3D->pts_board_2_obj_[boardid_charid] = pts_count;
              newObject3D->pts_obj_2_board_.push_back(boardid_charid);
              newObject3D->pts_3d_.push_back(trans_pts[k]);
              pts_count++;
            }
          }
        }
      }
    }

    // Add the 3D object into the structure
    object_3d[i] = newObject3D;
  }

  // Update the 3D object
  object_3d_.clear();
  object_3d_ = object_3d;
}

/**
 * @brief Merge the object observation
 *
 */
void Calibration::mergeAllObjectObs() {

  // First we erase all the object observation in the entire datastructure
  for (const auto &it : cams_) {
    it.second->vis_object_idx_.clear();
    it.second->object_observations_.clear();
  }

  for (const auto &it : cam_group_) {
    it.second->vis_object_idx_.clear();
    it.second->object_observations_.clear();
  }

  for (const auto &it : cams_group_obs_) {
    it.second->object_idx_.clear();
    it.second->object_observations_.clear();
  }

  for (const auto &it : cams_obs_) {
    it.second->object_observations_.clear();
    it.second->object_idx_.clear();
  }

  for (const auto &it : frames_) {
    it.second->object_observations_.clear();
    it.second->objects_idx_.clear();
  }

  for (const auto &it : object_3d_)
    it.second->object_observations_.clear();

  object_observations_.clear();

  // Reinitialize all the 3D object
  for (const auto &it_object : object_3d_) {
    // Reinitialize all object obserations
    for (const auto &it : object_3d_)
      this->init3DObjectObs(it.first);
  }
}

/**
 * @brief Compute the reprojection error for each camera group
 *
 */
void Calibration::reproErrorAllCamGroup() {
  for (const auto &it : cam_group_)
    it.second->reproErrorCameraGroup();
}

/**
 * @brief Non-linear optimization of the camera pose in the groups, the pose
 * of the observed objects, and the pose of the boards in the 3D objects
 *
 */
void Calibration::refineAllCameraGroupAndObjects() {
  for (const auto &it : cam_group_)
    it.second->refineCameraGroupAndObjects(nb_iterations_);

  // Update the 3D objects
  for (const auto &it : object_3d_)
    it.second->updateObjectPts();

  // Update the object3D observation
  for (const auto &it : cams_group_obs_)
    it.second->updateObjObsPose();
}

/**
 * @brief Save reprojection results images for a given camera.
 *
 */
void Calibration::saveReprojection(const int cam_id) {
  // Prepare the path to save the images
  std::string path_root = save_path_ + "Reprojection/";
  std::stringstream ss;
  ss << std::setw(3) << std::setfill('0') << cam_id;
  std::string cam_folder = ss.str();
  std::string path_save = path_root + cam_folder + "/";

  // check if the file exist and create it if it does not
  if (!boost::filesystem::exists(path_root) && path_root.length() > 0) {
    boost::filesystem::create_directories(path_root);
  }
  if (!boost::filesystem::exists(path_save) && path_root.length() > 0) {
    boost::filesystem::create_directory(path_save);
  }

  std::shared_ptr<Camera> cam = cams_[cam_id];

  // Iterate through the frames where this camera has visibility
  for (const auto &it_frame : frames_) {
    // Open the image
    std::string im_path = it_frame.second->frame_path_[cam_id];
    cv::Mat image = cv::imread(im_path);

    // Iterate through the camera group observations
    std::map<int, std::weak_ptr<CameraGroupObs>> cam_group_obs =
        it_frame.second->cam_group_observations_;
    for (const auto &it_cam_group_obs : cam_group_obs) {
      auto it_cam_group_obs_ptr = it_cam_group_obs.second.lock();
      if (it_cam_group_obs_ptr) {
        // Iterate through the object observation
        std::map<int, std::weak_ptr<Object3DObs>> object_observations =
            it_cam_group_obs_ptr->object_observations_;
        for (const auto &it_obj_obs : object_observations) {
          auto it_obj_obs_ptr = it_obj_obs.second.lock();
          auto it_obj_obs_cam_group_ptr =
              it_cam_group_obs_ptr->cam_group_.lock();
          auto it_obj_obs_object_3d_ptr = it_obj_obs_ptr->object_3d_.lock();
          if (it_obj_obs_ptr && it_obj_obs_cam_group_ptr &&
              it_obj_obs_object_3d_ptr &&
              it_obj_obs_ptr->camera_id_ == cam_id) {
            // Prepare the transformation matrix
            cv::Mat cam_pose =
                it_obj_obs_cam_group_ptr->getCameraPoseMat(cam_id) *
                it_obj_obs_ptr->getPoseInGroupMat();
            cv::Mat rot_vec, trans_vec;
            Proj2RT(cam_pose, rot_vec, trans_vec);

            // Get the 2d and 3d pts
            const std::vector<cv::Point2f> &pts_2d = it_obj_obs_ptr->pts_2d_;
            const std::vector<int> &pts_ind = it_obj_obs_ptr->pts_id_;
            const std::vector<cv::Point3f> &pts_3d_obj =
                it_obj_obs_object_3d_ptr->pts_3d_;
            std::vector<cv::Point2f> pts_repro;
            std::vector<cv::Point3f> pts_3d;
            pts_3d.reserve(pts_ind.size());
            for (const auto &pt_ind : pts_ind)
              pts_3d.emplace_back(pts_3d_obj[pt_ind]);

            // Reproject the pts
            cv::Mat rr, tt;
            rot_vec.copyTo(rr);
            trans_vec.copyTo(tt);
            projectPointsWithDistortion(pts_3d, rr, tt, cam->getCameraMat(),
                                        cam->getDistortionVectorVector(),
                                        pts_repro, cam->distortion_model_);

            // plot the keypoints on the image (red project // green detected)
            std::vector<double> color_repro{0, 0, 255};
            std::vector<double> color_detect{0, 255, 0};
            for (int i = 0; i < pts_2d.size(); i++) {
              cv::circle(
                  image, cv::Point(pts_repro[i].x, pts_repro[i].y), 4,
                  cv::Scalar(color_repro[0], color_repro[1], color_repro[2]),
                  cv::FILLED, 8, 0);
              cv::circle(
                  image, cv::Point(pts_2d[i].x, pts_2d[i].y), 4,
                  cv::Scalar(color_detect[0], color_detect[1], color_detect[2]),
                  cv::FILLED, 8, 0);
            }
          }
        }
      }
    }

    if (!image.empty()) {
      // display image
      // cv::imshow("reprojection_error", image);
      // cv::waitKey(1);

      // Save image
      std::stringstream ss1;
      ss1 << std::setw(6) << std::setfill('0') << it_frame.second->frame_idx_;
      std::string image_name = ss1.str() + ".jpg";
      cv::imwrite(path_save + image_name, image);
    }
  }
}

/**
 * @brief Save reprojection images for all camera
 *
 */
void Calibration::saveReprojectionAllCam() {
  for (const auto &it : cams_)
    saveReprojection(it.second->cam_idx_);
}

/**
 * @brief Save detection results images for a given camera
 *
 */
void Calibration::saveDetection(const int cam_id) {
  // Prepare the path to save the images
  std::string path_root = save_path_ + "Detection/";
  std::stringstream ss;
  ss << std::setw(3) << std::setfill('0') << cam_id;
  std::string cam_folder = ss.str();
  std::string path_save = path_root + cam_folder + "/";

  // check if the file exist and create it if it does not
  if (!boost::filesystem::exists(path_root) && path_root.length() > 0) {
    boost::filesystem::create_directories(path_root);
  }
  if (!boost::filesystem::exists(path_save) && path_root.length() > 0) {
    boost::filesystem::create_directory(path_save);
  }

  std::shared_ptr<Camera> cam = cams_[cam_id];

  // Iterate through the frames where this camera has visibility
  for (const auto &it_frame : frames_) {
    // Open the image
    std::string im_path = it_frame.second->frame_path_[cam_id];
    cv::Mat image = cv::imread(im_path);

    // Iterate through the camera group observations
    std::map<int, std::weak_ptr<CameraGroupObs>> cam_group_obs =
        it_frame.second->cam_group_observations_;
    for (const auto &it_cam_group_obs : cam_group_obs) {
      auto it_cam_group_obs_ptr = it_cam_group_obs.second.lock();
      if (it_cam_group_obs_ptr) {
        // Iterate through the object observation
        std::map<int, std::weak_ptr<Object3DObs>> object_observations =
            it_cam_group_obs_ptr->object_observations_;
        for (const auto &it_obj_obs : object_observations) {
          auto it_obj_obs_ptr = it_obj_obs.second.lock();
          auto it_obj_obs_object_3d_ptr = it_obj_obs_ptr->object_3d_.lock();
          if (it_obj_obs_ptr && it_obj_obs_ptr->camera_id_ == cam_id &&
              it_obj_obs_object_3d_ptr) {
            // Get the 2d and 3d pts
            const std::vector<cv::Point2f> &pts_2d = it_obj_obs_ptr->pts_2d_;
            // plot the keypoints on the image (red project // green detected)
            std::array<int, 3> &color = it_obj_obs_object_3d_ptr->color_;
            for (const auto &pt_2d : pts_2d) {
              cv::circle(image, cv::Point(pt_2d.x, pt_2d.y), 4,
                         cv::Scalar(color[0], color[1], color[2]), cv::FILLED,
                         8, 0);
            }
          }
        }
      }
    }

    if (!image.empty()) {
      // display image
      // cv::imshow("detection results", image);
      // cv::waitKey(1);

      // Save image
      std::stringstream ss1;
      ss1 << std::setw(6) << std::setfill('0') << it_frame.second->frame_idx_;
      std::string image_name = ss1.str() + ".jpg";
      cv::imwrite(path_save + image_name, image);
    }
  }
}

/**
 * @brief Save detection images for all camera
 *
 */
void Calibration::saveDetectionAllCam() {
  for (const auto &it : cams_)
    saveDetection(it.second->cam_idx_);
}

/**
 * @brief Initialize the intrinsic parameters and board pose of the entire
 * system
 *
 */
void Calibration::initIntrinsic() {
  initializeCalibrationAllCam();
  estimatePoseAllBoards();
  if (fix_intrinsic_ == 0) {
    refineIntrinsicAndPoseAllCam();
  }
  computeReproErrAllBoard();
}

/**
 * @brief Calibrate 3D objects
 *
 */
void Calibration::calibrate3DObjects() {
  computeBoardsPairPose();
  initInterTransform(board_pose_pairs_, inter_board_transform_);
  initInterBoardsGraph();
  init3DObjects();
  initAll3DObjectObs();
  estimatePoseAllObjects();
  computeReproErrAllObject();
  refineAllObject3D();
  computeReproErrAllObject();
}

/**
 * @brief Calibrate Camera groups
 *
 */
void Calibration::calibrateCameraGroup() {
  computeCamerasPairPose();
  initInterTransform(camera_pose_pairs_, inter_camera_transform_);
  initInterCamerasGraph();
  initCameraGroup();
  initAllCameraGroupObs();
  computeAllObjPoseInCameraGroup();
  refineAllCameraGroupAndObjects();
}

/**
 * @brief Merge objects
 *
 */
void Calibration::merge3DObjects() {
  initInterCamGroupGraph();
  estimatePoseAllObjects();
  computeAllObjPoseInCameraGroup();
  computeObjectsPairPose();
  initInterTransform(object_pose_pairs_, inter_object_transform_);
  initInterObjectsGraph();
  this->reproErrorAllCamGroup();
  mergeObjects();
  mergeAllObjectObs();
  mergeAllCameraGroupObs();
  estimatePoseAllObjects();
  computeAllObjPoseInCameraGroup();
  refineAllCameraGroupAndObjects();
  this->reproErrorAllCamGroup();
}

/**
 * @brief Compute distance between std::vectors of 2D points
 *
 * @return list of distances between points
 */
cv::Mat Calibration::computeDistanceBetweenPoints(
    const std::vector<cv::Point2f> obj_pts_2d,
    const std::vector<cv::Point2f> repro_pts) {
  cv::Mat error_list;
  for (int i = 0; i < repro_pts.size(); i++) {
    float rep_err = std::sqrt(std::pow((obj_pts_2d[i].x - repro_pts[i].x), 2) +
                              std::pow((obj_pts_2d[i].y - repro_pts[i].y), 2));
    error_list.push_back(rep_err);
  }
  return error_list;
}

/**
 * @brief Compute average reprojection error per camera group, cameras, frames,
 * observations
 *
 * @return average reprojection error
 */
double Calibration::computeAvgReprojectionError() {
  cv::Mat frame_list;
  cv::Scalar total_avg_error_sum;
  int number_of_adds = 0;

  for (const auto &it : cam_group_) {
    int cam_group_idx = it.second->cam_group_idx_;
    std::shared_ptr<CameraGroup> cur_cam_group = it.second;

    // iterate through frames
    for (const auto &it_frame : cur_cam_group->frames_) {
      auto it_frame_ptr = it_frame.second.lock();
      if (it_frame_ptr) {
        cv::Mat camera_list;
        frame_list.push_back(it_frame_ptr->frame_idx_);

        // iterate through cameraGroupObs
        std::map<int, std::weak_ptr<CameraGroupObs>> current_cam_group_obs_vec =
            it_frame_ptr->cam_group_observations_;
        for (const auto &it_cam_group_obs : current_cam_group_obs_vec) {
          // check if the current group is the camera group of interest
          auto it_cam_group_obs_ptr = it_cam_group_obs.second.lock();
          if (it_cam_group_obs_ptr &&
              cam_group_idx == it_cam_group_obs_ptr->cam_group_idx_) {
            std::map<int, std::weak_ptr<Object3DObs>> current_obj3d_obs_vec =
                it_cam_group_obs_ptr->object_observations_;

            // iterate through 3D object obs
            for (const auto &it_obj3d : current_obj3d_obs_vec) {
              auto it_obj3d_ptr = it_obj3d.second.lock();
              auto it_obj3d_object_3d_ptr = it_obj3d_ptr->object_3d_.lock();
              auto it_obj3d_cam_ptr = it_obj3d_ptr->cam_.lock();
              if (it_obj3d_ptr && it_obj3d_object_3d_ptr && it_obj3d_cam_ptr) {
                int current_cam_id = it_obj3d_ptr->camera_id_;
                const std::vector<cv::Point3f> &obj_pts_3d =
                    it_obj3d_object_3d_ptr->pts_3d_;
                const std::vector<int> &obj_pts_idx = it_obj3d_ptr->pts_id_;
                const std::vector<cv::Point2f> &obj_pts_2d =
                    it_obj3d_ptr->pts_2d_;
                camera_list.push_back(current_cam_id);

                // compute the reprojection error
                std::vector<cv::Point3f> object_pts;
                for (const auto &obj_pt_idx : obj_pts_idx)
                  object_pts.push_back(obj_pts_3d[obj_pt_idx]);

                // apply object pose transform
                std::vector<cv::Point3f> object_pts_trans1 =
                    transform3DPts(object_pts,
                                   it_cam_group_obs_ptr->getObjectRotVec(
                                       it_obj3d_ptr->object_3d_id_),
                                   it_cam_group_obs_ptr->getObjectTransVec(
                                       it_obj3d_ptr->object_3d_id_));
                // reproject pts
                std::vector<cv::Point2f> repro_pts;
                std::shared_ptr<Camera> cam_ptr = it_obj3d_cam_ptr;
                projectPointsWithDistortion(
                    object_pts_trans1,
                    it.second->getCameraRotVec(current_cam_id),
                    it.second->getCameraTransVec(current_cam_id),
                    cam_ptr->getCameraMat(),
                    cam_ptr->getDistortionVectorVector(), repro_pts,
                    cam_ptr->distortion_model_);

                cv::Mat error_list =
                    computeDistanceBetweenPoints(obj_pts_2d, repro_pts);
                total_avg_error_sum += cv::mean(error_list);
                number_of_adds++;
              }
            }
          }
        }
      }
    }
  }

  return total_avg_error_sum.val[0] / number_of_adds;
}

/**
 * @brief Save reprojection error in a file for analysis
 *
 */
void Calibration::saveReprojectionErrorToFile() {
  std::string save_reprojection_error =
      save_path_ + "reprojection_error_data.yml";
  cv::FileStorage fs(save_reprojection_error, cv::FileStorage::WRITE);
  cv::Mat frame_list;
  int nb_cam_group = cam_group_.size();
  fs << "nb_camera_group" << nb_cam_group;

  for (const auto &it : cam_group_) {
    int cam_group_idx = it.second->cam_group_idx_;
    fs << "camera_group_" + std::to_string(cam_group_idx);
    fs << "{";
    std::shared_ptr<CameraGroup> cur_cam_group = it.second;
    // iterate through frames
    for (const auto &it_frame : cur_cam_group->frames_) {
      std::shared_ptr<Frame> it_frame_ptr = it_frame.second.lock();
      if (it_frame_ptr) {
        cv::Mat camera_list;
        fs << "frame_" + std::to_string(it_frame_ptr->frame_idx_);
        fs << "{";
        frame_list.push_back(it_frame_ptr->frame_idx_);

        // iterate through cameraGroupObs
        std::map<int, std::weak_ptr<CameraGroupObs>> current_cam_group_obs_vec =
            it_frame_ptr->cam_group_observations_;
        for (const auto &it_cam_group_obs : current_cam_group_obs_vec) {
          auto it_cam_group_obs_ptr = it_cam_group_obs.second.lock();
          // check if the current group is the camera group of interest
          if (it_cam_group_obs_ptr &&
              cam_group_idx == it_cam_group_obs_ptr->cam_group_idx_) {
            std::map<int, std::weak_ptr<Object3DObs>> current_obj3d_obs_vec =
                it_cam_group_obs_ptr->object_observations_;

            // iterate through 3D object obs
            for (const auto &it_obj3d : current_obj3d_obs_vec) {
              auto it_obj3d_ptr = it_obj3d.second.lock();
              auto it_obj3d_object_3d_ptr = it_obj3d_ptr->object_3d_.lock();
              auto it_cam_group_obs_ptr = it_cam_group_obs.second.lock();
              auto it_obj3d_cam_ptr = it_obj3d_ptr->cam_.lock();
              if (it_obj3d_ptr && it_obj3d_object_3d_ptr &&
                  it_cam_group_obs_ptr && it_obj3d_cam_ptr) {
                int current_cam_id = it_obj3d_ptr->camera_id_;
                const std::vector<cv::Point3f> &obj_pts_3d =
                    it_obj3d_object_3d_ptr->pts_3d_;
                const std::vector<int> &obj_pts_idx = it_obj3d_ptr->pts_id_;
                const std::vector<cv::Point2f> &obj_pts_2d =
                    it_obj3d_ptr->pts_2d_;
                camera_list.push_back(current_cam_id);
                fs << "camera_" + std::to_string(current_cam_id);
                fs << "{";

                // compute the reprojection error
                std::vector<cv::Point3f> object_pts;
                for (const auto &obj_pt_idx : obj_pts_idx)
                  object_pts.push_back(obj_pts_3d[obj_pt_idx]);

                int nb_pts = obj_pts_idx.size();
                fs << "nb_pts" << nb_pts;

                // apply object pose transform
                std::vector<cv::Point3f> object_pts_trans1 =
                    transform3DPts(object_pts,
                                   it_cam_group_obs_ptr->getObjectRotVec(
                                       it_obj3d_ptr->object_3d_id_),
                                   it_cam_group_obs_ptr->getObjectTransVec(
                                       it_obj3d_ptr->object_3d_id_));
                // reproject pts
                std::vector<cv::Point2f> repro_pts;
                std::shared_ptr<Camera> cam_ptr = it_obj3d_cam_ptr;
                projectPointsWithDistortion(
                    object_pts_trans1,
                    it.second->getCameraRotVec(current_cam_id),
                    it.second->getCameraTransVec(current_cam_id),
                    cam_ptr->getCameraMat(),
                    cam_ptr->getDistortionVectorVector(), repro_pts,
                    cam_ptr->distortion_model_);

                cv::Mat error_list =
                    computeDistanceBetweenPoints(obj_pts_2d, repro_pts);
                fs << "error_list" << error_list << "}";
              }
            }
          }
        }
        fs << "camera_list" << camera_list << "}";
      }
    }
    fs << "frame_list" << frame_list << "}";
  }
  fs.release();
}

/**
 * @brief Non-linear optimization of the camera pose in the groups, the pose
 * of the observed objects, and the pose of the boards in the 3D objects
 *
 */
void Calibration::refineAllCameraGroupAndObjectsAndIntrinsic() {
  for (const auto &it : cam_group_)
    it.second->refineCameraGroupAndObjectsAndIntrinsics(nb_iterations_);

  // Update the 3D objects
  for (const auto &it : object_3d_)
    it.second->updateObjectPts();

  // Update the object3D observation
  for (const auto &it : cams_group_obs_)
    it.second->updateObjObsPose();
}
