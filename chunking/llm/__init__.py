"""LLM extraction package.

Importing this package injects the operating system's certificate trust store
into Python's TLS stack (best-effort). The ``openai``/``httpx`` clients used by
our providers otherwise only trust the ``certifi`` CA bundle, which fails behind
a TLS-intercepting proxy/VPN whose CA lives in the OS store but not in certifi
(symptom: ``APIConnectionError`` / ``CERTIFICATE_VERIFY_FAILED``). truststore
makes httpx trust the same CAs the OS does, matching ``urllib`` behavior.
"""

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    # truststore is optional: if it's missing or injection fails, fall back to
    # certifi's bundle. Installs without a TLS-intercepting proxy are unaffected.
    pass
