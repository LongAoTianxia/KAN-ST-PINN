# 二维波动方程有限差分求解器（吸收边界）
# 参考 Langtangen《Finite Difference Computing with PDEs》
# 代码: https://github.com/hplgit/fdm-book.git


from tqdm import tqdm
import numpy as np

def solver(I, c, Lx, Ly, Nx, Ny, dt, T, dtype=np.float32):
    
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy =  np.meshgrid(x, y)

    c_max = np.max(c(xx, yy))
    stability_limit = (1/float(c_max))*(1/np.sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print('error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit))
        
    Nt = int(np.ceil(T/dt)+1)
    t = np.linspace(0, (Nt-1)*dt, Nt)
    dt2 = dt**2
    dx2 = dx**2
    dy2 = dy**2

    # 初始化数组
    u     = np.zeros((Nx,Ny))
    u_n   = np.zeros((Nx,Ny))
    u_nm1 = np.zeros((Nx,Ny))
    u_log = np.zeros((Nx,Ny,Nt))
    It = range(0, t.shape[0]+1)
    q_a = c(xx, yy)**2

    # 加载初始条件
    u_n[:,:] = I(xx, yy)
    u_log[:,:,0] = u_n
    
    # 第一步时间推进
    n = 0
    u = advance(
        u, u_n, u_nm1, q_a, dx2, dy2, dt2, step1=True)

    # 更新下一步
    u_nm1, u_n, u = u_n, u, u_nm1

    for n in tqdm(It[1:-1]):
        u = advance(u, u_n, u_nm1, q_a, dx2, dy2, dt2)

        # 更新下一步
        u_log[:,:,n] = u
        u_nm1, u_n, u = u_n, u, u_nm1
        
    return u_log.astype(dtype),\
            xx.astype(dtype),\
            yy.astype(dtype),\
            t.astype(dtype),\
            dt.astype(dtype)

def advance(u, u_n, u_nm1, q_a, dx2, dy2, dt2, step1=False):
    if step1:
        D1, D2, D3 = 1, 0, 0.5
    else:
        D1, D2, D3 = 2, 1, 1

    u[1:-1,1:-1] = D1*u_n[1:-1,1:-1] - D2*u_nm1[1:-1,1:-1] + \
          D3*( ((q_a[1:-1,1:-1] + q_a[2:,1:-1])*(u_n[2:,1:-1] - u_n[1:-1,1:-1]) - \
                (q_a[1:-1,1:-1] + q_a[:-2,1:-1])*(-u_n[:-2,1:-1] + u_n[1:-1,1:-1]))*dt2/(2*dx2) + \
               ((q_a[1:-1,1:-1] + q_a[1:-1,2:])*(u_n[1:-1,2:] - u_n[1:-1,1:-1]) - \
                (q_a[1:-1,1:-1] + q_a[1:-1,:-2])*(-u_n[1:-1,:-2] + u_n[1:-1,1:-1]))*dt2/(2*dy2) )
    
    # 边界条件
    j = 0
    cdt = np.sqrt(q_a[:,j]) * np.sqrt(dt2)
    beta = (cdt - np.sqrt(dy2)) / (cdt + np.sqrt(dy2))
    u[:,j] = u_n[:,j+1] + beta*(u[:,j+1] - u_n[:,j])

    j = u.shape[1]-1
    cdt = np.sqrt(q_a[:,j]) * np.sqrt(dt2)
    beta = (cdt - np.sqrt(dy2)) / (cdt + np.sqrt(dy2))
    u[:,j] = u_n[:,j-1] + beta*(u[:,j-1] - u_n[:,j])

    i = 0
    cdt = np.sqrt(q_a[i,:]) * np.sqrt(dt2)
    beta = (cdt - np.sqrt(dx2)) / (cdt + np.sqrt(dx2))
    u[i,:] = u_n[i+1,:] + beta*(u[i+1,:] - u_n[i,:])
    
    i = u.shape[0]-1
    cdt = np.sqrt(q_a[i,:]) * np.sqrt(dt2)
    beta = (cdt - np.sqrt(dx2)) / (cdt + np.sqrt(dx2))
    u[i,:] = u_n[i-1,:] + beta*(u[i-1,:] - u_n[i,:])
    
    return u