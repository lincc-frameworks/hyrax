def test_train(loopback_hyrax):
    """
    Simple test that training succeeds with the loopback
    model in use.
    """
    h, _ = loopback_hyrax
    h.train()
