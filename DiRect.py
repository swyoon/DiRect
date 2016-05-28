import numpy as np

class DiRect(object):
    def __init__(self, f, bounds, epsilon=None, max_feval=20, max_iter=10, minimize=True):
        self.f = f  # should take (D,) nparray as input
        self.bounds = bounds  # Dx2 [[lower, upper]]
        self.epsilon = epsilon
        self.max_feval = max_feval
        self.max_iter = max_iter
        self.minimize = minimize
        
        if epsilon is not None:
            raise NotImplementedError
        
        if not self.minimize:  # means maximization problem
            self.f_wrap = lambda x: -self.f(x)
        else:
            self.f_wrap = self.f

        self.scale = bounds[:,1] - bounds[:,0]
        self.shift = bounds[:,0]
        self.D = bounds.shape[0]
        
        self.curr_opt = np.inf
        self.x_at_opt = None
        self.x_at_opt_unit = None
        self.n_feval = 0
        self.n_iter = 0
        self.n_rectdiv = 0
        self.d_rect = {}        
        self.l_hist = []

        assert isinstance(bounds, np.ndarray)
        assert len(bounds.shape) == 2
        assert bounds.shape[1] == 2
        assert np.all(self.scale > 0.)
        ## TODO: other assertions

    def u2r(self, unit_coord):
        """unit to real: map a coordinate 
        in unit hypercube to one in the actual rectangle"""
        return unit_coord * self.scale + self.shift
    
    def true_sign(self, val):
        if self.minimize:
            return val
        else:
            return -val
        
    
    def divide_rectangle(self, po_rect):
        D = self.D
        maxlen = np.max(po_rect.sides)
        gap = maxlen / 3.
        d_new_rects = {}
        
        self.d_rect[po_rect.d2].remove(po_rect)
        
        # only the longest sides are divided
        maxlen_sides = list(np.nonzero(po_rect.sides == maxlen)[0])
        
        # evaluate points near center
        for side_idx in maxlen_sides:
            d_new_rects[side_idx] = []
            
            new_center_u = po_rect.center.copy()
            new_center_u[side_idx] += gap
            new_fval_u = self.f_wrap(self.u2r(new_center_u))
            d_new_rects[side_idx].append(Rectangle(new_center_u, new_fval_u, po_rect.sides.copy()))
            self.l_hist.append((self.u2r(new_center_u), self.true_sign(new_fval_u)))
            self.n_feval += 1
            if new_fval_u < self.curr_opt:
                self.curr_opt = new_fval_u
                self.x_at_opt_unit = new_center_u.copy()
                self.x_at_opt = self.u2r(self.x_at_opt_unit)
            if self.n_feval > self.max_feval:
                self.TERMINATE = True
                return
            
            new_center_l = po_rect.center.copy()
            new_center_l[side_idx] -= gap
            new_fval_l = self.f_wrap(self.u2r(new_center_l))
            d_new_rects[side_idx].append(Rectangle(new_center_l, new_fval_l, po_rect.sides.copy()))
            self.l_hist.append((self.u2r(new_center_l), self.true_sign(new_fval_l)))
            self.n_feval += 1
            if new_fval_l < self.curr_opt:
                self.curr_opt = new_fval_l
                self.x_at_opt = new_center_l.copy()
                self.x_at_opt = self.u2r(self.x_at_opt_unit)
            if self.n_feval > self.max_feval:
                self.TERMINATE = True
                return
        
        # axis with better function value get divided first
        maxlen_sides = sorted(maxlen_sides, 
                              key=lambda x: min([t.f_val for t in d_new_rects[x]]))
        
        for i in xrange(len(maxlen_sides)):
            for side_idx in maxlen_sides[i:]:
                po_rect.sides[side_idx] /= 3.  # po_rect gets divided in every (longest) dimension
                for each_rect in d_new_rects[side_idx]:
                    each_rect.sides[side_idx] /= 3.
                    
        
        for l_rect in d_new_rects.values():
            for each_rect in l_rect:
                d2 = each_rect.d2 
                if d2 not in self.d_rect:
                    self.d_rect[d2] = [each_rect]
                else:
                    if each_rect.f_val < self.d_rect[d2][0].f_val:
                        self.d_rect[d2].insert(0, each_rect)
                    else:
                        self.d_rect[d2].append(each_rect)
        
        # insert po_rect
        if po_rect.d2 not in self.d_rect:
            self.d_rect[po_rect.d2] = [po_rect]
        else:
            if po_rect.f_val < self.d_rect[po_rect.d2][0].f_val:
                self.d_rect[po_rect.d2].insert(0, po_rect)
            else:
                self.d_rect[po_rect.d2].append(po_rect)
        
        # remove empty list from the dictionary
        for dd in [key for key in self.d_rect if len(self.d_rect[key]) == 0]:
            self.d_rect.pop(dd)

    def get_potentially_optimal_rects(self):
        # among rects with the same size, choose the one with the smallest function value
        border = [(key, l[0].f_val) for key, l in self.d_rect.items()]
        border = sorted(border, key=lambda t:t[0])
        
        l_po_key = []
        for i in xrange(len(border)-1):
            if border[i][1] <= border[i+1][1]:
                l_po_key.append(border[i][0])
        l_po_key.append(border[-1][0])
            
            
        return [self.d_rect[key][0] for key in l_po_key]
            
    def run(self):
        D = self.D
        
        # initialize
        c = np.array([0.5] * D)
        f_val = self.f_wrap(self.u2r(c))
        s = np.array([1.]*D)
        rect = Rectangle(c, f_val, s)
        
        self.d_rect[rect.d2] = [rect]
        self.l_hist.append((self.u2r(c), self.true_sign(f_val)))
        self.n_feval += 1
        self.curr_opt = f_val
        self.x_at_opt_unit = c
        self.x_at_opt = self.u2r(c)
        self.TERMINATE = False
        
        for i in xrange(self.max_iter):            
            # select potentially optimal rectangles
#             l_potentially_optimal = [l[0] for l in self.d_rect.values()]
            l_potentially_optimal = self.get_potentially_optimal_rects()
            
            for po_rect in l_potentially_optimal:
                self.divide_rectangle(po_rect)
                
                if self.TERMINATE:
                    break
            if self.TERMINATE:
                break
                
        return self.true_sign(self.curr_opt), self.x_at_opt, self.l_hist
    
class Rectangle(object):
    def __init__(self, center, f_val, sides):
        self.center = center
        self.f_val = f_val
        self.sides = sides
        self.__str__ = f_val
        
    @property
    def d2(self):
        return np.sum((self.sides / 2.) ** 2)