import threading
import traceback
from robotutils.base.print_tools import print_red
import links_and_nodes as ln
import numpy as np


class ArmCollisionChecker_LnInterface:
    def __init__(self, ln_clnt=None, name="ArmCollisionCheckerService"):
        if ln_clnt is not None:
            self.ln_clnt = ln_clnt
        else:
            self.ln_clnt = ln.client(name, [])
        self.services = {}
        self.services["svc.collision_check"] = self.ln_clnt.get_service(
            f"{name}.collision_check", "python_modules/environment_awareness/arm_collision_avoidance/collision_check_svc"
        )
        self.services["svc.reset_octomap"] = self.ln_clnt.get_service(f"{name}.reset_octomap", "navigation/reset_map_svc")
        self.elbow_attractor = self.ln_clnt.subscribe(
            "arm_collision_avoidance.elbow_attractor",
            "python_modules/environment_awareness/arm_collision_avoidance/elbow_attractor",
        )


    def check_collisionfree_target(self, target: np.ndarray):
        """
        returns:
            - target ee collisionfree : bool | None (None if collision check is disabled)
            - last valid ee pose : np.array 4x4 (target itself if collision check is disabled)
        """
        try:
            svc = self.services["svc.collision_check"]
            svc.req.ee_target = target.flatten()
            svc.call()
            if svc.resp.error_message != "":
                raise Exception(svc.resp.error_message)

            free = None if svc.resp.free < 0 else bool(svc.resp.free)  # -1 -> collision check disabled
            last_valid = np.array(svc.resp.last_valid).reshape(4, 4) if len(svc.resp.last_valid) else np.array([])
            return free, last_valid
        except Exception as e:
            print_red(f"[Error] Arm collision checker not responding or something went wrong")
            print(e)
            return None, None


    def get_elbow_attractor_point(self):
        """
        returns:
            - elbow attractor position : np.array 3x1 | None (None if inactive/no data)
        """
        self.elbow_attractor.read(-1)
        if not bool(self.elbow_attractor.packet.active):
            return None
        return np.asarray(self.elbow_attractor.packet.position, dtype=float).reshape(3, 1)


    def reset_octomap(self):
        try:
            svc = self.services["svc.reset_octomap"]
            svc.req.trigger = 1
            svc.call()
            if svc.resp.error_message != "":
                raise Exception(svc.resp.error_message)
        except Exception as e:
            print_red(f"[Error] Arm collision checker not responding or something went wrong")
            print(e)
            return None


class ServiceProvider(object):
    def __init__(self, collision_manager, name="ArmCollisionCheckerService"):
        self.ln_clnt = collision_manager.ln_clnt
        self.manager = collision_manager
        self._reset_octomap_thread = None
        svc1 = self.ln_clnt.get_service_provider(
            f"{name}.collision_check",
            "python_modules/environment_awareness/arm_collision_avoidance/collision_check_svc",
        )
        svc1.set_handler(self.svc_collision_check)
        svc1.do_register()

        svc4 = self.ln_clnt.get_service_provider(
            f"{name}.reset_octomap",
            "navigation/reset_map_svc",
        )
        svc4.set_handler(self.svc_reset_octomap)
        svc4.do_register()


    def _respond(self, svc, req, resp):
        svc.respond()
        resp.result = None
        resp.error_message = ""
        resp.error_type = ""

    def svc_collision_check(self, svc, req, resp) -> None:
        try:
            assert len(req.ee_target) == 16
            #print("[ArmCollisionCheckerService] svc.collision_check - Service Call received")
            target = np.array(req.ee_target).reshape(4, 4)

            free = self.manager.check_collisionfree_target(target)
            resp.free = int(free)

            if not resp.free:
                last_valid = self.manager.get_last_valid()
                if last_valid is not None:
                    resp.last_valid = last_valid.flatten()

            else:
                resp.last_valid = target

            print(f"[ArmCollisionCheckerService] svc.collision_check - Desired ee pose free?: {resp.free}")

        except Exception:
            ex = traceback.format_exc()
            resp.error_message = ex
        finally:
            self._respond(svc, req, resp)


    def svc_reset_octomap(self, svc, req, resp) -> None:
        try:
            print("[ArmCollisionCheckerService] svc.reset_octomap - Service Call received")
            if self.manager.pin_robot.use_octomap:
                if self._reset_octomap_thread is not None and self._reset_octomap_thread.is_alive():
                    print("[ArmCollisionCheckerService] svc.reset_octomap - reset already running, skipping")
                    return
                # run in a separate thread so the ln loop keeps handling
                # other service requests (eg collision_check) while resetting
                self._reset_octomap_thread = threading.Thread(
                    target=self._reset_octomap,
                    name="reset_octomap",
                    daemon=True,
                )
                self._reset_octomap_thread.start()
        except Exception:
            ex = traceback.format_exc()
            resp.error_message = ex
        finally:
            self._respond(svc, req, resp)

    def _reset_octomap(self) -> None:
        try:
            success = self.manager.pin_robot.octree_getter.reset_octomap()
            print(f"[ArmCollisionCheckerService] octomap reset {'successful' if success else 'failed'}")
        except Exception:
            print(traceback.format_exc())