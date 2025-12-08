import torch

def get_surface_laplacian(model, v_cart, n):
    def model_scalar(v_):
        return model(v_).view(-1).requires_grad_(True)  # Ensure scalar output

    f_ = model_scalar(v_cart)
    
    grad_f = torch.autograd.grad(f_, v_cart, torch.ones_like(f_), create_graph=True, retain_graph=True)[0]

    # Surface gradient (tangential component)
    grad_f_surf = grad_f - torch.sum(grad_f * n, dim=1, keepdim=True) * n

    # Compute surface divergence of surface gradient
    def compute_divergence(grad_, v_):
        # Compute derivatives of each component of grad_f_surf
        div_x = torch.autograd.grad(grad_[:, 0], v_, torch.ones_like(grad_[:, 0]), create_graph=True, retain_graph=True)[0]
        div_x = div_x - torch.sum(div_x * n, dim=1, keepdim=True) * n
        div_y = torch.autograd.grad(grad_[:, 1], v_, torch.ones_like(grad_[:, 1]), create_graph=True, retain_graph=True)[0]
        div_y = div_y - torch.sum(div_y * n, dim=1, keepdim=True) * n
        div_z = torch.autograd.grad(grad_[:, 2], v_, torch.ones_like(grad_[:, 2]), create_graph=True, retain_graph=True)[0]
        div_z = div_z - torch.sum(div_z * n, dim=1, keepdim=True) * n
        
        # Build Hessian matrix
        hessian = torch.zeros(v_cart.shape[0], 3, 3).to(v_cart.device)
        hessian[:, 0, :] = div_x
        hessian[:, 1, :] = div_y
        hessian[:, 2, :] = div_z

        # Sum diagonal terms for divergence
        divF = hessian[:, 0, 0] + hessian[:, 1, 1] + hessian[:, 2, 2]

        return divF, hessian

    div_grad_f_surf, hessians = compute_divergence(grad_f_surf, v_cart)

    # Compute Hessian applied to normal: H n
    hessian_dot_n = torch.bmm(hessians, n.unsqueeze(-1)).squeeze()

    # Compute normal term: n^T (H n)
    normals_term = torch.sum(n * hessian_dot_n, dim=1)

    # Final Laplace-Beltrami operator
    lap_beltrami = div_grad_f_surf - normals_term

    return lap_beltrami
