import cv2
import numpy as np

class ColorCheckerMonitor:
    def __init__(self, roi, min_inliers=10, inlier_ratio_threshold=0.3, move_threshold=5.0):
        """
        初始化校色标记监测器

        参数:
            roi: 校色标记区域 (x, y, w, h)
            min_inliers: 最小内点数量阈值
            inlier_ratio_threshold: 内点比例阈值
            move_threshold: 移动阈值(像素)
        """
        print("初始化校色标记监测器", roi)
        self.roi = roi
        self.min_inliers = min_inliers
        self.inlier_ratio_threshold = inlier_ratio_threshold
        self.move_threshold = move_threshold

        # 计算参考标记中心点（基于ROI）
        x, y, w, h = roi
        self.ref_center = np.array([x + w//2, y + h//2], dtype=np.float32)
        self.prev_center = self.ref_center.copy()

        # 初始化特征检测器和匹配器
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 参考帧相关属性（稍后初始化）
        self.ref_roi = None
        self.ref_kp = None
        self.ref_desc = None
        self.initialized = False  # 标记是否完成初始化

    def _initialize_reference(self, frame_gray):
        """使用当前帧初始化参考特征"""
        x, y, w, h = self.roi
        self.ref_roi = frame_gray[y:y+h, x:x+w]

        # 提取参考特征
        self.ref_kp, self.ref_desc = self.detector.detectAndCompute(self.ref_roi, None)

        if self.ref_desc is None or len(self.ref_kp) < 4:
            print("警告: 初始化参考帧失败，特征点不足，等待下一帧")
            return False

        print(f"参考帧初始化成功! 检测到 {len(self.ref_kp)} 个特征点")
        self.initialized = True
        return True

    def check_movement(self, frame):
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.roi
        cur_roi = gray[y:y+h, x:x+w]

        # 尚未初始化参考帧：尝试初始化
        if not self.initialized:
            success = self._initialize_reference(gray)
            # 无论是否成功初始化，当前帧不视为移动
            return False, 0.0, None

        # 检测当前帧特征点
        cur_kp, cur_desc = self.detector.detectAndCompute(cur_roi, None)

        # 特征点不足处理
        if cur_desc is None or len(cur_kp) < 4:
            # print("警告: 当前帧特征点不足，保留参考帧")
            return True, float('inf'), None

        # 匹配特征点
        matches = self.matcher.match(self.ref_desc, cur_desc)

        # 匹配点不足处理
        if len(matches) < 4:
            # print(f"警告: 匹配点不足({len(matches)}个)，更新参考帧")
            self._update_reference(cur_kp, cur_desc)
            return True, float('inf'), None

        # 准备点对进行单应性计算
        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        cur_pts = np.float32([cur_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(ref_pts, cur_pts, cv2.RANSAC, 5.0)

        # 统计内点数量
        inliers = mask.ravel().sum() if mask is not None else 0
        inlier_ratio = inliers / len(matches) if len(matches) > 0 else 0

        # 检查匹配质量
        if inliers < self.min_inliers or inlier_ratio < self.inlier_ratio_threshold:
            # print(f"警告: 匹配质量差(内点:{inliers}/{len(matches)}，比例:{inlier_ratio:.2f})，更新参考帧")
            self._update_reference(cur_kp, cur_desc)
            return True, float('inf'), None

        # 计算中心点偏移
        ref_center_roi = np.array([self.ref_roi.shape[1]//2, self.ref_roi.shape[0]//2])
        cur_center_roi = cv2.perspectiveTransform(
            ref_center_roi.reshape(1, 1, 2).astype(np.float32), H
        )[0][0] if H is not None else ref_center_roi

        # 计算实际图像坐标中的偏移
        cur_center = np.array([x + cur_center_roi[0], y + cur_center_roi[1]])
        frame_offset = np.linalg.norm(cur_center - self.prev_center)

        # 判断是否移动
        moved = frame_offset > self.move_threshold

        # 更新参考为当前帧（用于下一帧比较）
        self._update_reference(cur_kp, cur_desc)
        self.prev_center = cur_center.copy()

        return moved, frame_offset, cur_center

    def _update_reference(self, kp, desc):
        """更新参考帧特征"""
        self.ref_kp = kp
        self.ref_desc = desc