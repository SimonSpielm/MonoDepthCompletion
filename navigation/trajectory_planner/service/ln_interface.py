import json

import links_and_nodes as ln


class TrajectoryPlaner_LnInterface:
    def __init__(self, ln_clnt=None, name="TrajectoryPlanner"):
        if ln_clnt is not None:
            self.ln_clnt = ln_clnt
        else:
            self.ln_clnt = ln.client(name, [])
        self.services = {}
        self.services["svc.sct_commands"] = self.ln_clnt.get_service(
            f"{name}.sct_commands", "navigation/sct_commands"
        )

    def send_sct_commands(self, commands: dict):
        svc = self.services["svc.sct_commands"]
        svc.req.commands = json.dumps(commands)
        svc.call()
        if svc.resp.error_message != "":
            raise Exception(svc.resp.error_message)
        print("called trajectory planer")
