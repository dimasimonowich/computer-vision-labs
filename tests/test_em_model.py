from models.em_model import EM
import pytest
import numpy as np


@pytest.mark.parametrize("eps", [0.1, 0.01, 1, 10])
def test_em_params_eps_true(eps):
    em = EM(eps=eps)
    assert em.eps == eps


@pytest.mark.parametrize("eps", [0, -0.1, -0.01, -1, -10])
def test_em_params_eps_false(eps):
    with pytest.raises(AssertionError):
        EM(eps=eps)


@pytest.mark.parametrize("num_epoch", [1, 5, 100, 1000])
def test_em_params_num_epoch_true(num_epoch):
    em = EM(num_epoch=num_epoch)
    assert em.num_epoch == num_epoch


@pytest.mark.parametrize("num_epoch", [0, -1, -0.1, 1.1])
def test_em_params_num_epoch_false(num_epoch):
    with pytest.raises(AssertionError):
        EM(num_epoch=num_epoch)


@pytest.mark.parametrize("num_classes", [2, 5, 100, 1000])
def test_em_params_num_epoch_true(num_classes):
    em = EM(num_classes=num_classes)
    assert em.num_classes == num_classes


@pytest.mark.parametrize("num_classes", [0, 1, -1, -0.1, 1.1])
def test_em_params_num_epoch_false(num_classes):
    with pytest.raises(AssertionError):
        EM(num_classes=num_classes)


@pytest.mark.parametrize("X, p_classes_Xm", [(np.zeros((2, 2)),
                                              np.array([[0.5488135, 0.71518937],
                                                        [0.60276338, 0.54488318]])
                                              ),
                                             (np.zeros((3, 5)),
                                              np.array([[0.5488135, 0.71518937, 0.60276338],
                                                        [0.54488318, 0.4236548, 0.64589411]]))])
def test_em_initialize_probs(X, p_classes_Xm):
    np.random.seed(0)
    em = EM()
    em.initialize_probs(X)
    assert np.allclose(em.p_classes_Xm, p_classes_Xm)


@pytest.mark.parametrize("X, p_classes_Xm, p_ij_classes", [(np.array([[0], [1], [1]]),
                                                            np.array([[0.5, 0.5, 0.5],
                                                                      [0.5, 0.5, 0.5]]),
                                                            np.array([[0.66666667], [0.66666667]]))])
def test_em_perform_maximization_step(X, p_classes_Xm, p_ij_classes):
    em = EM()
    em.p_classes_Xm = p_classes_Xm
    em.perform_maximization_step(X)
    assert np.allclose(em.p_ij_classes, p_ij_classes)


@pytest.mark.parametrize("X, p_classes_Xm, p_classes",    [(np.array([[0], [1], [1]]),
                                                            np.array([[0.2, 0.2, 0.5],
                                                                      [0.8, 0.8, 0.5]]),
                                                            np.array([[0.3], [0.7]]))])
def test_em_update_p_classes(X, p_classes_Xm, p_classes):
    em = EM()
    em.p_classes_Xm = p_classes_Xm
    em.update_p_classes(X)
    assert np.allclose(em.p_classes, p_classes)


@pytest.mark.parametrize("X, p_classes_Xm, expected_p_classes_Xm", [(np.array([[0], [1], [1]]),
                                                                     np.array([[0.2, 0.2, 0.5],
                                                                               [0.8, 0.8, 0.5]]),
                                                                     np.array([[0.2, 0.35, 0.35],
                                                                               [0.8, 0.65, 0.65]]))])
def test_em_perform_expectation_step(X, p_classes_Xm, expected_p_classes_Xm):
    em = EM()
    em.p_classes_Xm = p_classes_Xm
    em.perform_maximization_step(X)
    em.update_p_classes(X)
    em.perform_expectation_step(X)
    assert np.allclose(em.p_classes_Xm, expected_p_classes_Xm)


@pytest.mark.parametrize("X", [np.array([[0, 1],
                                         [1, 0]]),
                               np.array([[0, 0, 1],
                                         [1, 0, 0]])])
def test_em_fit_predict(X):
    em = EM()
    prediction = em.fit_predict(X)
    assert prediction[0] != prediction[1]

