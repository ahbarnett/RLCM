#ifndef _ISOTROPIC_GAUSSIAN_TPP_
#define _ISOTROPIC_GAUSSIAN_TPP_


//--------------------------------------------------------------------------
template<class Point>
double IsotropicGaussian::
Eval(const Point &x, const Point &y, double lambda) const {
  if (x.GetD() != y.GetD()) {
    printf("IsotropicGaussian::Eval. Error: Point dimensions do not match. Return NAN.\n");
    return NAN;
  }
  double r2 = x.Dist2(y);
  double ret = s * exp(-r2/Square(sigma)/2.0);
  if (r2 == 0.0) {
    ret += lambda;
  }
  return ret;
}


#endif
