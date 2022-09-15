from ..scripts.regression.Example_3_LSTM_with_dropout import main
import pytest


@pytest.mark.skip(reason="need manual action")
def test_main_script_run():

    main()
    assert True