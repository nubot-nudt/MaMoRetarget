# import os
# import time
# import mujoco as mj
# import mujoco.viewer as mjv
# import imageio
# from scipy.spatial.transform import Rotation as R
# from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
# from loop_rate_limiters import RateLimiter
# import numpy as np
# from rich import print


# def draw_frame(
#     pos,
#     mat,
#     v,
#     size,
#     joint_name=None,
#     orientation_correction=R.from_euler("xyz", [0, 0, 0]),
#     pos_offset=np.array([0, 0, 0]),
# ):
#     def _set_if(obj, name, value):
#         if hasattr(obj, name):
#             setattr(obj, name, value)

#     rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

#     for i in range(3):
#         geom = v.user_scn.geoms[v.user_scn.ngeom]
#         mj.mjv_initGeom(
#             geom,
#             mj.mjtGeom.mjGEOM_LINE,
#             np.zeros(3),
#             np.zeros(3),
#             np.zeros(9),
#             rgba_list[i],
#         )

#         # 基本字段（绝大多数版本都有）
#         _set_if(geom, "size", getattr(geom, "size", None))  # 仅确保存在
#         if hasattr(geom, "size"):
#             geom.size[0] = 2
#             geom.size[1] = 2

#         geom.pos = pos + pos_offset
#         geom.mat = mat @ orientation_correction.as_matrix()

#         start = geom.pos
#         end = geom.pos + mat[:, i] * size

#         # 下面这些字段在不同版本可能不存在，所以都保护一下
#         _set_if(geom, "dataid", 0)
#         _set_if(geom, "objtype", mj.mjtObj.mjOBJ_UNKNOWN)
#         _set_if(geom, "objid", -1)
#         _set_if(geom, "category", getattr(mj, "mjtCatBit", None).mjCAT_DECOR if hasattr(mj, "mjtCatBit") else 0)
#         _set_if(geom, "segid", 0)

#         # texture / material 相关字段：部分版本没有（你这里就没有 texid）
#         _set_if(geom, "texid", -1)
#         _set_if(geom, "texuniform", 0)
#         _set_if(geom, "emission", 0)
#         _set_if(geom, "specular", 0.5)
#         _set_if(geom, "shininess", 0.5)
#         _set_if(geom, "reflectance", 0)

#         # LINE 的 start/end 通常放在 geom.data 里（不同版本表现也可能不同）
#         if hasattr(geom, "data"):
#             try:
#                 geom.data = np.concatenate([start, end])
#             except Exception:
#                 # 有些绑定里 geom.data 是只读数组，只能逐元素写
#                 try:
#                     arr = np.concatenate([start, end]).astype(float)
#                     for k in range(min(len(arr), len(geom.data))):
#                         geom.data[k] = float(arr[k])
#                 except Exception:
#                     pass

#         v.user_scn.ngeom += 1

#     # 可选：显示 label（不同版本不一定支持）
#     if joint_name is not None:
#         geom = v.user_scn.geoms[v.user_scn.ngeom]
#         mj.mjv_initGeom(
#             geom,
#             mj.mjtGeom.mjGEOM_LABEL,
#             np.zeros(3),
#             np.zeros(3),
#             np.zeros(9),
#             [1, 1, 1, 1],
#         )

#         if hasattr(geom, "size"):
#             geom.size[0] = 0.5
#             geom.size[1] = 0.5
#         geom.pos = pos + pos_offset
#         geom.mat = mat

#         _set_if(geom, "dataid", 0)
#         _set_if(geom, "objtype", mj.mjtObj.mjOBJ_UNKNOWN)
#         _set_if(geom, "objid", -1)
#         _set_if(geom, "category", getattr(mj, "mjtCatBit", None).mjCAT_DECOR if hasattr(mj, "mjtCatBit") else 0)
#         _set_if(geom, "segid", 0)

#         # label 字段在你版本里也可能没有
#         _set_if(geom, "label", joint_name)

#         v.user_scn.ngeom += 1


# class RobotMotionViewer_warn:
#     def __init__(self,
#                 robot_type,
#                 camera_follow=True,
#                 motion_fps=30,
#                 transparent_robot=0,
#                 # video recording
#                 record_video=False,
#                 video_path=None,
#                 video_width=640,
#                 video_height=480):

#         self.robot_type = robot_type
#         self.xml_path = ROBOT_XML_DICT[robot_type]
#         self.model = mj.MjModel.from_xml_path(str(self.xml_path))
#         self.data = mj.MjData(self.model)

#         self.robot_base = ROBOT_BASE_DICT[robot_type]
#         self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]

#         self.motion_fps = motion_fps
#         self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
#         self.camera_follow = camera_follow
#         self.record_video = record_video

#         self.viewer = mjv.launch_passive(
#             model=self.model,
#             data=self.data,
#             show_left_ui=False,
#             show_right_ui=False)

#         self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

#         # --- cache original visualization state for collision highlighting ---
#         self._orig_geom_rgba = self.model.geom_rgba.copy()
#         self._world_body_id = self.model.body("world").id if hasattr(self.model, "body") else 0
#         self._nonworld_geom_mask = (self.model.geom_bodyid != self._world_body_id)

#         # cache contact visualization flags (so we can restore user's defaults)
#         try:
#             self._orig_show_contact_point = int(self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT])
#             self._orig_show_contact_force = int(self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE])
#         except Exception:
#             self._orig_show_contact_point = 0
#             self._orig_show_contact_force = 0

#         if self.record_video:
#             assert video_path is not None, "Please provide video path for recording"
#             self.video_path = video_path
#             video_dir = os.path.dirname(self.video_path)

#             if not os.path.exists(video_dir):
#                 os.makedirs(video_dir)
#             self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
#             print(f"Recording video to {self.video_path}")

#             # Initialize renderer for video recording
#             self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)

#     def _compute_self_contacts(self, exclude_world: bool = True):
#         """Return a list of robot internal contacts for the current frame.
#         Each item includes body/geom names and signed distance (contact.dist).
#         """
#         contacts = []
#         model, data = self.model, self.data

#         for ci in range(data.ncon):
#             c = data.contact[ci]
#             g1, g2 = c.geom1, c.geom2
#             if g1 < 0 or g2 < 0:
#                 continue

#             b1 = model.geom_bodyid[g1]
#             b2 = model.geom_bodyid[g2]

#             body1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
#             body2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)

#             if exclude_world and (body1 == "world" or body2 == "world"):
#                 continue

#             geom1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1)
#             geom2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2)

#             contacts.append({
#                 "geom1": geom1,
#                 "geom2": geom2,
#                 "body1": body1,
#                 "body2": body2,
#                 "dist": float(c.dist),
#                 "normal": c.frame[0:3].copy(),
#             })

#         return contacts

#     def _set_collision_visual(self, on: bool, n_contacts: int = 0):
#         """Make collision frames visually obvious (robot turns red, contact points/forces shown)."""
#         if on:
#             # Tint non-world geoms red for strong visual cue.
#             self.model.geom_rgba[self._nonworld_geom_mask, :3] = np.array([1.0, 0.1, 0.1])
#             self.model.geom_rgba[self._nonworld_geom_mask, 3] = 1.0

#             # Show contact points/forces (very helpful to locate the collision).
#             try:
#                 self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
#                 self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
#             except Exception:
#                 pass

#             # Optional overlay text (depends on mujoco version/viewer build).
#             try:
#                 self.viewer.add_overlay(
#                     mj.mjtGridPos.mjGRID_TOPLEFT,
#                     "⚠ SELF-COLLISION",
#                     f"{n_contacts} contacts"
#                 )
#             except Exception:
#                 pass
#         else:
#             # Restore original appearance and flags.
#             self.model.geom_rgba[:] = self._orig_geom_rgba
#             try:
#                 self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = self._orig_show_contact_point
#                 self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = self._orig_show_contact_force
#             except Exception:
#                 pass

#     def step(self,
#             # robot data
#             root_pos, root_rot, dof_pos,
#             # human data
#             human_motion_data=None,
#             show_human_body_name=False,
#             # scale for human point visualization
#             human_point_scale=0.1,
#             # human pos offset add for visualization
#             human_pos_offset=np.array([0.0, 0.0, 0]),
#             # rate limit
#             rate_limit=True,
#             follow_camera=True,
#             # collision highlighting
#             highlight_self_collision=False,
#             return_self_collision_contacts=False,
#             ):
#         """
#         by default visualize robot motion.
#         also support visualize human motion by providing human_motion_data, to compare with robot motion.

#         human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

#         if rate_limit is True, the motion will be visualized at the same rate as the motion data.
#         else, the motion will be visualized as fast as possible.
#         """

#         self.data.qpos[:3] = root_pos
#         self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
#         self.data.qpos[7:] = dof_pos

#         mj.mj_forward(self.model, self.data)

#         if follow_camera:
#             self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
#             self.viewer.cam.distance = self.viewer_cam_distance
#             self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
#             # self.viewer.cam.azimuth = 180    # 正面朝向机器人

#         # ===== self-collision detection & highlighting (before sync so it shows up immediately) =====
#         frame_contacts = None
#         if highlight_self_collision or return_self_collision_contacts:
#             frame_contacts = self._compute_self_contacts(exclude_world=True)
#             is_collision = (len(frame_contacts) > 0)
#             if highlight_self_collision:
#                 self._set_collision_visual(is_collision, n_contacts=len(frame_contacts))
#         else:
#             is_collision = False

#         if human_motion_data is not None:
#             # Clean custom geometry
#             self.viewer.user_scn.ngeom = 0
#             # Draw the task targets for reference
#             for human_body_name, (pos, rot) in human_motion_data.items():
#                 draw_frame(
#                     pos,
#                     R.from_quat(rot, scalar_first=True).as_matrix(),
#                     self.viewer,
#                     human_point_scale,
#                     pos_offset=human_pos_offset,
#                     joint_name=human_body_name if show_human_body_name else None
#                     )

#         self.viewer.sync()
#         if rate_limit is True:
#             self.rate_limiter.sleep()

#         if self.record_video:
#             # Use renderer for proper offscreen rendering
#             self.renderer.update_scene(self.data, camera=self.viewer.cam)
#             img = self.renderer.render()
#             self.mp4_writer.append_data(img)

#         if return_self_collision_contacts:
#             return frame_contacts
#         return None

#     def close(self):
#         self.viewer.close()
#         time.sleep(0.5)
#         if self.record_video:
#             self.mp4_writer.close()
#             print(f"Video saved to {self.video_path}")


import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    """
    Robust human-frame drawing across MuJoCo Python bindings.

    Why this version:
    - Some mujoco bindings omit fields like `texid` (and sometimes even make `geom.data` absent/readonly),
      which breaks "manual LINE with geom.data" drawing.
    - `mj.mjv_connector(...)` is more stable: it fills the underlying geom representation for arrows/lines
      without requiring us to touch version-specific struct fields.

    This mirrors the working approach in robot_motion_viewer.py (mjv_connector-based). 
    """
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    fix = orientation_correction.as_matrix()

    # draw 3 axes as arrows
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]

        # init as an arrow (size is used as arrowhead scaling; connector defines the segment)
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )

        # Optional label support (not all bindings have `label`)
        if joint_name is not None and hasattr(geom, "label"):
            geom.label = joint_name

        start = pos + pos_offset
        end = start + size * (mat @ fix)[:, i]

        # connector writes the arrow segment robustly
        mj.mjv_connector(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=start,
            to=end,
        )

        v.user_scn.ngeom += 1


class RobotMotionViewer_warn:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480):

        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)

        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]

        # Do one forward pass so camera/lookat has valid xpos even before first external step.
        mj.mj_forward(self.model, self.data)

        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False)

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        # --- cache original visualization state for collision highlighting ---
        self._orig_geom_rgba = self.model.geom_rgba.copy()
        self._world_body_id = self.model.body("world").id if hasattr(self.model, "body") else 0
        self._nonworld_geom_mask = (self.model.geom_bodyid != self._world_body_id)

        # cache contact visualization flags (so we can restore user's defaults)
        try:
            self._orig_show_contact_point = int(self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT])
            self._orig_show_contact_force = int(self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE])
        except Exception:
            self._orig_show_contact_point = 0
            self._orig_show_contact_force = 0

        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)

            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")

            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)

    def _compute_self_contacts(self, exclude_world: bool = True):
        """Return a list of robot internal contacts for the current frame.
        Each item includes body/geom names and signed distance (contact.dist).
        """
        contacts = []
        model, data = self.model, self.data

        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 < 0 or g2 < 0:
                continue

            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]

            body1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            body2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)

            if exclude_world and (body1 == "world" or body2 == "world"):
                continue

            geom1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1)
            geom2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2)

            contacts.append({
                "geom1": geom1,
                "geom2": geom2,
                "body1": body1,
                "body2": body2,
                "dist": float(c.dist),
                "normal": c.frame[0:3].copy(),
            })

        return contacts

    def _set_collision_visual(self, on: bool, n_contacts: int = 0):
        """Make collision frames visually obvious (robot turns red, contact points/forces shown)."""
        if on:
            # Tint non-world geoms red for strong visual cue.
            self.model.geom_rgba[self._nonworld_geom_mask, :3] = np.array([1.0, 0.1, 0.1])
            self.model.geom_rgba[self._nonworld_geom_mask, 3] = 1.0

            # Show contact points/forces (very helpful to locate the collision).
            try:
                self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            except Exception:
                pass

            # Optional overlay text (depends on mujoco version/viewer build).
            try:
                self.viewer.add_overlay(
                    mj.mjtGridPos.mjGRID_TOPLEFT,
                    "⚠ SELF-COLLISION",
                    f"{n_contacts} contacts"
                )
            except Exception:
                pass
        else:
            # Restore original appearance and flags.
            self.model.geom_rgba[:] = self._orig_geom_rgba
            try:
                self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = self._orig_show_contact_point
                self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = self._orig_show_contact_force
            except Exception:
                pass

    def step(self,
            # robot data
            root_pos, root_rot, dof_pos,
            # human data
            human_motion_data=None,
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True,
            follow_camera=True,
            # collision highlighting
            highlight_self_collision=False,
            return_self_collision_contacts=False,
            ):
        """
        Visualize robot motion, optionally overlay human motion (axes) and highlight self-collisions.

        human_motion_data: dict[str, (pos(3,), rot_quat_wxyz(4,))], where rot is wxyz (scalar_first=True).
        """

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot  # wxyz
        self.data.qpos[7:] = dof_pos

        mj.mj_forward(self.model, self.data)

        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # front-ish view

        # ===== self-collision detection & highlighting (before sync so it shows up immediately) =====
        frame_contacts = None
        if highlight_self_collision or return_self_collision_contacts:
            frame_contacts = self._compute_self_contacts(exclude_world=True)
            is_collision = (len(frame_contacts) > 0)
            if highlight_self_collision:
                self._set_collision_visual(is_collision, n_contacts=len(frame_contacts))

        # ===== human overlay =====
        if human_motion_data is not None:
            # IMPORTANT: this only clears *custom* geoms; it won't affect MuJoCo's own contact viz.
            self.viewer.user_scn.ngeom = 0

            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None,
                )

        self.viewer.sync()

        if rate_limit:
            self.rate_limiter.sleep()

        if self.record_video:
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)

        if return_self_collision_contacts:
            return frame_contacts
        return None

    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")


