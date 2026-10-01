# Service authentication

All HTTP routes require exactly one `x-backend-auth` header matching the server's `BACKEND_SECRET`. Missing server configuration fails closed. Only exact GET/HEAD `/health` and `/healthz` are public; the middleware returns minimal health without invoking application handlers. WebSocket requests require the same service authentication. Keep the key on trusted servers, never in a browser or mobile application. Provider callback credentials are separate from this inbound key.

This carries forward the authentication middleware already deployed during the September 29 incident. Existing legacy clients remain paused until they supply the configured service credential. This change does not authorize restoring an unauthenticated route or sharing the service key with end users.

Run `python -m unittest -v test_service_auth test_security_contract` before release. These tests use synthetic secrets and do not call providers, databases, or customer callbacks. They must also run during image build.

The repository entrypoint differs from the captured production source. Do not deploy this entire checkout over production until those differences and the exact locked dependencies have been reviewed. Current immutable production containment must remain in place. This source integration does not certify the remaining application routes, outbound requests, dependencies, or callback ownership.
