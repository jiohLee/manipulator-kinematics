import numpy as np

class Kinematic:
    def __init__(self, dh_params, in_workspace=lambda x, dh: True):
        self.dh_params = dh_params
        self.theta = None
        self.in_workspace = in_workspace

    def transform(self, theta):
        M = self.dh_params.shape[0]
        N = theta.shape[0]

        length = self.dh_params[:, 0]
        offset = self.dh_params[:, 1]
        twist = self.dh_params[:, 2]

        c, s = np.cos(theta), np.sin(theta)
        ct, st = np.cos(twist), np.sin(twist)

        A = np.stack([np.eye(4)] * N)
        
        A[..., 0, 0] = c
        A[..., 0, 1] = -s * ct
        A[..., 0, 2] = s * st
        A[..., 0, 3] = length * c

        A[..., 1, 0] = s
        A[..., 1, 1] = c * ct
        A[..., 1, 2] = -c * st
        A[..., 1, 3] = length * s

        A[..., 2, 1] = st
        A[..., 2, 2] = ct
        A[..., 2, 3] = offset

        return A

    def forward(self, theta):
        A = self.transform(theta)
        return np.linalg.multi_dot(A)[..., :3, 3]
    
    def path(self, theta):
        A = self.transform(theta)
        N = theta.shape[0]

        T = np.stack([np.eye(4)] * (N + 1))
        for i in range(1, N + 1):
            T[i] = T[i - 1] @ A[i - 1]

        return T[..., :3, 3]
    
    def inverse(self, x_des, theta, lam=0.1, nu=10, step_sz=1, max_iter=1000, tol=1e-5, verbose=False):
        """
        TODO:
        * 긴 구간을 왕복해서 이동할 경우, ellbow 위치가 바뀌는 현상이 생김(branch switching이 일어남). 어떻게 해결?
            -> 해결 방법을 모색하거나, 다른 라이브러리를 쓰거나...
        * link 끼리 서로 교차하는 singularity가 존재. 어떻게 해결?
            -> 미리 '경로 계획' 하여 해당 궤도에서 운동할 때 관절각을 일정하게?
        """

        if not self.in_workspace(x_des, self.dh_params):
            return self.forward(theta), theta

        return Kinematic._damped_least_square(self.forward, x_des, theta, 
                                         lam, nu, step_sz, 
                                         max_iter, tol, verbose=verbose)
    
    @staticmethod
    def _central_difference_jacobian(f, x, eps=1e-5):
        x = np.asarray(x, dtype=float)
        fx = f(x)
        n, m = x.size, fx.size
        J = np.zeros([m, n])

        for i in range(n):
            dx = np.zeros_like(x)
            dx[i] = eps
            f_plus = f(x + dx)
            f_minus = f(x - dx)
            J[:, i] = (f_plus - f_minus) / (2 * eps)

        return J

    @staticmethod
    def _damped_least_square(f, x_des, theta, lam=0.1, nu=10, step_sz=1, max_iter=1000, tol=1e-5, verbose=False):
        N = theta.shape[0]

        x = f(theta)
        for i in range(max_iter):
            dx = x_des - x
            prv_err = np.linalg.norm(dx)
            if prv_err < tol:
                break

            J = Kinematic._central_difference_jacobian(f, theta)
            a = J.T @ J + (lam ** 2) * np.eye(J.shape[1])
            b = J.T @ dx
            d_theta = np.linalg.inv(a) @ b

            cur_theta = theta + step_sz * d_theta
            cur_x = f(cur_theta)
            cur_err = np.linalg.norm(x_des - cur_x)
            
            if verbose:
                print(i, prv_err, cur_err)
            
            if cur_err < prv_err:
                theta = cur_theta
                x = cur_x
                lam /= nu
            else:
                lam *= nu

        return x, theta


if __name__ == "__main__":
    
    dh_params = np.array([
        (15 * np.sqrt(2), 0, 0),
        (15 * np.sqrt(2), 0, 0),
    ])
    km = Kinematic(dh_params=dh_params)

    theta = np.deg2rad([45, 90])

    T = km.forward(theta)
    print(f"with theta {np.rad2deg(theta)}, end-effector point at: {T}")

    T = km.path(np.deg2rad([45, 90]))
    print(f"with theta {np.rad2deg(theta)}, path to the end-effector point:\n{T}")

    x_des = np.array([0, 30, 0])
    theta = np.array([0, 0])
    print(f"desired end-effector point {x_des}, and current theta {np.rad2deg(theta)}")

    x, theta = km.inverse(x_des, theta, verbose=True)
    print(f"estimated theta {theta}(rad), {np.rad2deg(theta)}(deg) at end-effector point at: {x}")
    print(f"error: {np.linalg.norm(x_des - x)}")

    theta = np.deg2rad([30, -30])
    x_des = np.array([42, 0, 0])
    print(f"desired end-effector point {x_des}, and current theta {np.rad2deg(theta)}")

    x, theta = km.inverse(x_des, theta)
    print(f"estimated theta {theta}(rad), {np.rad2deg(theta)}(deg) at end-effector point at: {x}")
    print(f"error: {np.linalg.norm(x_des - x)}")
    
    for i in range(42, 10, -1):
        x_des = np.array([i, 0, 0])
        x, theta = km.inverse(x_des, theta)

        print(f"rad: {theta} deg: {np.rad2deg(theta)} pt: {x} error: {np.linalg.norm(x_des - x)}")
    