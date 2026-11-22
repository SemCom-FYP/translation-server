# coding: utf-8
"""
Simultaneous speech-to-speech translation server built on Meta's
SeamlessStreaming (github.com/facebookresearch/seamless_communication),
a standalone project: it shares no code or environment with LivePortrait or
webrtc_conference (they only talk over a WebSocket).

    server.py           the server (runs in its own Linux / WSL2 environment)
    client_adapter.py   client library; RemoteTranslation is a drop-in for
                        webrtc_conference.translator.PeerTranslation
    wire.py             the WebSocket protocol, shared by both sides

This file deliberately imports nothing: the client adapter must load in the
conference app's environment, which has neither the model code nor soxr.
"""

__version__ = "0.2.0"
