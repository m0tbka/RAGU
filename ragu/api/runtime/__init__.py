"""
The machinery around requests rather than inside them.

Authentication, admission and body limits (``middleware``, ``auth``), request
ids (``request_context``), background jobs (``jobs``), the graph catalogue
(``registry``), Prometheus counters (``metrics``) and the bridge that routes
stdlib logging into loguru (``logging_setup``).
"""
