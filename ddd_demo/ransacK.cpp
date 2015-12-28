#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <assert.h>
#include <limits>


#define comp_max(x,y) ((x)>(y)?(x):(y))
#define comp_min(x,y) ((x)<(y)?(x):(y))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

float gen_random_float(float min, float max) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(min, max - 0.0001);
  return dist(mt);
}

double PYTHAG(double a, double b)
{
  double at = fabs(a), bt = fabs(b), ct, result;

  if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
  else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
  else result = 0.0;
  return (result);
}

void dsvd(float a[4][4], int m, int n, float w[4], float v[4][4])
{
  /*
   * Input to dsvd is as follows:
   *   a = mxn matrix to be decomposed, gets overwritten with u
   *   m = row dimension of a
   *   n = column dimension of a
   *   w = returns the vector of singular values of a
   *   v = returns the right orthogonal transformation matrix
   *   http://www.public.iastate.edu/~dicook/JSS/paper/code/svd.c
  */
  int flag, i, its, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  double rv1[4];

  if (m < n)
  {
    printf("#rows must be > #cols \n");
    return;
  }
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < n; i++)
  {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m)
    {
      for (k = i; k < m; k++)
        scale += fabs((double)a[k][i]);
      if (scale)
      {
        for (k = i; k < m; k++)
        {
          a[k][i] = (float)((double)a[k][i] / scale);
          s += ((double)a[k][i] * (double)a[k][i]);
        }
        f = (double)a[i][i];
        //SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][i] = (float)(f - g);
        if (i != n - 1)
        {
          for (j = l; j < n; j++)
          {
            for (s = 0.0, k = i; k < m; k++)
              s += ((double)a[k][i] * (double)a[k][j]);
            f = s / h;
            for (k = i; k < m; k++)
              a[k][j] += (float)(f * (double)a[k][i]);
          }
        }
        for (k = i; k < m; k++)
          a[k][i] = (float)((double)a[k][i] * scale);
      }
    }
    w[i] = (float)(scale * g);

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != n - 1)
    {
      for (k = l; k < n; k++)
        scale += fabs((double)a[i][k]);
      if (scale)
      {
        for (k = l; k < n; k++)
        {
          a[i][k] = (float)((double)a[i][k] / scale);
          s += ((double)a[i][k] * (double)a[i][k]);
        }
        f = (double)a[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][l] = (float)(f - g);
        for (k = l; k < n; k++)
          rv1[k] = (double)a[i][k] / h;
        if (i != m - 1)
        {
          for (j = l; j < m; j++)
          {
            for (s = 0.0, k = l; k < n; k++)
              s += ((double)a[j][k] * (double)a[i][k]);
            for (k = l; k < n; k++)
              a[j][k] += (float)(s * rv1[k]);
          }
        }
        for (k = l; k < n; k++)
          a[i][k] = (float)((double)a[i][k] * scale);
      }
    }
    anorm = comp_max(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = n - 1; i >= 0; i--)
  {
    if (i < n - 1)
    {
      if (g)
      {
        for (j = l; j < n; j++)
          v[j][i] = (float)(((double)a[i][j] / (double)a[i][l]) / g);
        /* double division to avoid underflow */
        for (j = l; j < n; j++)
        {
          for (s = 0.0, k = l; k < n; k++)
            s += ((double)a[i][k] * (double)v[k][j]);
          for (k = l; k < n; k++)
            v[k][j] += (float)(s * (double)v[k][i]);
        }
      }
      for (j = l; j < n; j++)
        v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = n - 1; i >= 0; i--)
  {
    l = i + 1;
    g = (double)w[i];
    if (i < n - 1)
      for (j = l; j < n; j++)
        a[i][j] = 0.0;
    if (g)
    {
      g = 1.0 / g;
      if (i != n - 1)
      {
        for (j = l; j < n; j++)
        {
          for (s = 0.0, k = l; k < m; k++)
            s += ((double)a[k][i] * (double)a[k][j]);
          f = (s / (double)a[i][i]) * g;
          for (k = i; k < m; k++)
            a[k][j] += (float)(f * (double)a[k][i]);
        }
      }
      for (j = i; j < m; j++)
        a[j][i] = (float)((double)a[j][i] * g);
    }
    else
    {
      for (j = i; j < m; j++)
        a[j][i] = 0.0;
    }
    ++a[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = n - 1; k >= 0; k--)
  { /* loop over singular values */
    for (its = 0; its < 30; its++)
    { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--)
      { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm)
        {
          flag = 0;
          break;
        }
        if (fabs((double)w[nm]) + anorm == anorm)
          break;
      }
      if (flag)
      {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++)
        {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm)
          {
            g = (double)w[i];
            h = PYTHAG(f, g);
            w[i] = (float)h;
            h = 1.0 / h;
            c = g * h;
            s = (- f * h);
            for (j = 0; j < m; j++)
            {
              y = (double)a[j][nm];
              z = (double)a[j][i];
              a[j][nm] = (float)(y * c + z * s);
              a[j][i] = (float)(z * c - y * s);
            }
          }
        }
      }
      z = (double)w[k];
      if (l == k)
      { /* convergence */
        if (z < 0.0)
        { /* make singular value nonnegative */
          w[k] = (float)(-z);
          for (j = 0; j < n; j++)
            v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        //free((void*) rv1);
        printf("No convergence after 30,000! iterations \n");
        return;
      }

      /* shift from bottom 2 x 2 minor */
      x = (double)w[l];
      nm = k - 1;
      y = (double)w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++)
      {
        i = j + 1;
        g = rv1[i];
        y = (double)w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < n; jj++)
        {
          x = (double)v[jj][j];
          z = (double)v[jj][i];
          v[jj][j] = (float)(x * c + z * s);
          v[jj][i] = (float)(z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = (float)z;
        if (z)
        {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++)
        {
          y = (double)a[jj][j];
          z = (double)a[jj][i];
          a[jj][j] = (float)(y * c + z * s);
          a[jj][i] = (float)(z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = (float)x;
    }
  }
  //free((void*) rv1);
  return;
}

void quat2rot(const float Q[4], float R[9])
{
  /*  QUAT2ROT */
  /*    R = QUAT2ROT(Q) converts a quaternion (4x1 or 1x4) into a rotation mattrix */
  R[0] = ((Q[0] * Q[0] + Q[1] * Q[1]) - Q[2] * Q[2]) - Q[3] * Q[3];
  R[1] = 2.0 * (Q[1] * Q[2] - Q[0] * Q[3]);
  R[2] = 2.0 * (Q[1] * Q[3] + Q[0] * Q[2]);
  R[3] = 2.0 * (Q[1] * Q[2] + Q[0] * Q[3]);
  R[4] = ((Q[0] * Q[0] - Q[1] * Q[1]) + Q[2] * Q[2]) - Q[3] * Q[3];
  R[5] = 2.0 * (Q[2] * Q[3] - Q[0] * Q[1]);
  R[6] = 2.0 * (Q[1] * Q[3] - Q[0] * Q[2]);
  R[7] = 2.0 * (Q[2] * Q[3] + Q[0] * Q[1]);
  R[8] = ((Q[0] * Q[0] - Q[1] * Q[1]) - Q[2] * Q[2]) + Q[3] * Q[3];
  return;
}

void crossTimesMatrix(const float V[9], int V_length, float*** V_times)
{
  //V a 3xN matrix, rpresenting a series of 3x1 vectors
  for (int i = 0; i < V_length; i++) {
    V_times[0][0][i] = 0;
    V_times[0][1][i] = -V[2 + 3 * i];
    V_times[0][2][i] =  V[1 + 3 * i];

    V_times[1][0][i] =  V[2 + 3 * i];
    V_times[1][1][i] = 0;
    V_times[1][2][i] = -V[0 + 3 * i];

    V_times[2][0][i] = -V[1 + 3 * i];
    V_times[2][1][i] =  V[0 + 3 * i];
    V_times[2][2][i] = 0;
  }
  return;
}
void transpose4by4(float a[4][4], float b[4][4]) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      b[j][i] = a[i][j];
    }
  }
  return;
}
void multi4by4 (float a[4][4], float b[4][4], float c[4][4]) {
  for (int i = 0; i < 4 ; i++) {
    for (int j = 0 ; j < 4 ; j++) {
      c[i][j] = 0;
      for (int k = 0 ; k < 4; k++) {
        c[i][j] = c[i][j] + a[i][k] * b[k][j];
      }
    }
  }
  return;
}


void estimateRigidTransform(const float* x_in, const float* y_in, const int pointCount, float RT[12], bool debug = 0)
//void estimateRigidTransform(const float* h_coord, const int* h_randPts,  int idx, int numLoops,float* RT)
{
  // for 3 point only
  // X: [x0,y0,z0,x1,y1,z1,x2,y2,z2] : 1x9
  // Y: [x0',y0',z0',x1,y1',z1',x2',y2',z2'] :1x9
  // xh = T * yh

  // form the 3x3 point matrix
  //int pointCount = 3;
  //bool debug = 1;


  float x_centroid[3] = {0.0, 0.0, 0.0};
  float y_centroid[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < pointCount; i++) {
    x_centroid[0] += x_in[i * 3];
    x_centroid[1] += x_in[i * 3 + 1];
    x_centroid[2] += x_in[i * 3 + 2];
    y_centroid[0] += y_in[i * 3];
    y_centroid[1] += y_in[i * 3 + 1];
    y_centroid[2] += y_in[i * 3 + 2];
  }


  for (int i = 0; i < 3; i++) {
    x_centroid[i] = x_centroid[i] / pointCount;
    y_centroid[i] = y_centroid[i] / pointCount;
  }

  //printf("x_centroid:\n");
  //printf("%f,%f,%f\n",y_centroid[0],y_centroid[1],y_centroid[2]);

  float* x = new float[3 * pointCount];
  float* y = new float[3 * pointCount];
  for (int i = 0; i < pointCount; i++) {
    x[0 + i * 3] = x_in[0 + i * 3] - x_centroid[0];
    x[1 + i * 3] = x_in[1 + i * 3] - x_centroid[1];
    x[2 + i * 3] = x_in[2 + i * 3] - x_centroid[2];

    y[0 + i * 3] = y_in[0 + i * 3] - y_centroid[0];
    y[1 + i * 3] = y_in[1 + i * 3] - y_centroid[1];
    y[2 + i * 3] = y_in[2 + i * 3] - y_centroid[2];

  }
  // printf("x\n");
  // for (int jj =0; jj<3;jj++){
  //   printf("%f,%f,%f\n",x[0+jj*3],x[1+3*jj],x[2+3*jj]);
  // }
  //y_centrized = y
  float* R12 = new float[3 * pointCount];
  for  (int i = 0; i < pointCount; i++) {
    R12[0 + i * 3] = y[0 + i * 3] - x[0 + i * 3];
    R12[1 + i * 3] = y[1 + i * 3] - x[1 + i * 3];
    R12[2 + i * 3] = y[2 + i * 3] - x[2 + i * 3];
  }

  float* R21 = new float[3 * pointCount];
  for  (int i = 0; i < pointCount; i++) {
    R21[0 + i * 3] = - y[0 + i * 3] + x[0 + i * 3];
    R21[1 + i * 3] = - y[1 + i * 3] + x[1 + i * 3];
    R21[2 + i * 3] = - y[2 + i * 3] + x[2 + i * 3];
  }

  float* R22_1 = new float[3 * pointCount];
  for  (int i = 0; i < pointCount; i++) {
    R22_1[0 + i * 3] = y[0 + i * 3] + x[0 + i * 3];
    R22_1[1 + i * 3] = y[1 + i * 3] + x[1 + i * 3];
    R22_1[2 + i * 3] = y[2 + i * 3] + x[2 + i * 3];
  }

  if (debug) {
    printf("R12\n");
    for (int jj = 0; jj < pointCount; jj++) {
      printf("%f,%f,%f\n", R12[0 + jj * 3], R12[1 + 3 * jj], R12[2 + 3 * jj]);
    }
    printf("R22_1\n");
    for (int jj = 0; jj < pointCount; jj++) {
      printf("%f,%f,%f\n", R22_1[0 + jj * 3], R22_1[1 + 3 * jj], R22_1[2 + 3 * jj]);
    }
  }






  //float R22[3][3][3];
  float*** R22 = new float**[3];
  for (int i = 0; i < 3; i++) {
    R22[i] = new float*[3];
    for (int j = 0; j < 3; j++) {
      R22[i][j] = new float[pointCount];
    }
  }

  crossTimesMatrix(R22_1, pointCount, R22);
  float B[4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      B[i][j] = 0;
    }
  }


  float A[4][4];
  for  (int i = 0; i < pointCount; i++) {
    A[0][0] = 0;
    A[0][1] = R12[0 + i * 3];
    A[0][2] = R12[1 + i * 3];
    A[0][3] = R12[2 + i * 3];

    A[1][0] = R21[0 + i * 3];
    A[1][1] = R22[0][0][i];
    A[1][2] = R22[0][1][i];
    A[1][3] = R22[0][2][i];

    A[2][0] = R21[1 + i * 3];
    A[2][1] = R22[1][0][i];
    A[2][2] = R22[1][1][i];
    A[2][3] = R22[1][2][i];

    A[3][0] = R21[2 + i * 3];
    A[3][1] = R22[2][0][i];
    A[3][2] = R22[2][1][i];
    A[3][3] = R22[2][2][i];

    float A_p[4][4];
    transpose4by4(A, A_p);
    float AA_p[4][4];
    multi4by4(A, A_p, AA_p);

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        B[j][k] = B[j][k] + AA_p[j][k];
      }
    }
    /*
    printf("A%d\n",i);
    for (int jj =0; jj<4;jj++){
       printf("%f,%f,%f,%f\n",A[jj][0],A[jj][1],A[jj][2],A[jj][3]);
    }

    printf("A_p%d\n",i);
    for (int jj =0; jj<4;jj++){
      printf("%f,%f,%f,%f\n",AA_p[jj][0],AA_p[jj][1],AA_p[jj][2],AA_p[jj][3]);
    }

    printf("B%d\n",i);
    for (int jj =0; jj<4;jj++){
      printf("%f,%f,%f,%f\n",B[jj][0],B[jj][1],B[jj][2],B[jj][3]);
    }
    */

  }
  float S[4] = {0, 0, 0, 0};
  float V[4][4] = {0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0
                  };



  dsvd(B, 4, 4, S, V);

  int ind = 0;
  float minsingularvalue = S[0];
  for (int i = 0; i < 4; i++) {
    if (S[i] < minsingularvalue) {
      minsingularvalue = S[i];
      ind = i;
    }
  }
  float quat[4];
  for (int i = 0; i < 4; i++) {
    quat[i] = V[i][ind];
  }

  float rot[9];
  quat2rot(quat, rot);

  /*
  printf("V\n");
  for (int jj =0; jj<4;jj++){
  printf("%f,%f,%f,%f\n",V[0][jj],V[1][jj],V[2][jj],V[3][jj]);
  }
  printf("S\n");
  printf("%f,%f,%f,%f\n",S[0],S[1],S[2],S[3]);
  printf("quat :%f,%f,%f,%f\n",quat[0],quat[1],quat[2],quat[3]);
  printf("rot\n");
  for (int jj =0; jj<3;jj++){
  printf("%f,%f,%f\n",rot[0+jj*3],rot[1+3*jj],rot[2+3*jj]);
  }
  */


  float T1[4][4] = {1, 0, 0, -y_centroid[0],
                    0, 1, 0, -y_centroid[1],
                    0, 0, 1, -y_centroid[2],
                    0, 0, 0, 1
                   };
  float T2[4][4] = {rot[0], rot[1], rot[2], 0,
                    rot[3], rot[4], rot[5], 0,
                    rot[6], rot[7], rot[8], 0,
                    0, 0, 0, 1
                   };
  float T3[4][4] = {1, 0, 0, x_centroid[0],
                    0, 1, 0, x_centroid[1],
                    0, 0, 1, x_centroid[2],
                    0, 0, 0, 1
                   };


  float T21[4][4];
  multi4by4(T2, T1, T21);


  float T[4][4];
  multi4by4(T3, T21, T);



  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      RT[i * 4 + j] = T[i][j];
    }
  }


  /*
  printf("T\n");
  for (int jj =0; jj<4;jj++){
    printf("%f,%f,%f,%f\n",T[jj][0],T[jj][1],T[jj][2],T[jj][3]);
  }
  */

  delete[] x, y, R12, R21, R22_1;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      delete [] R22[i][j];
    }
    delete [] R22[i];
  }
  delete[] R22;


  return;
}

std::vector<int> TestRigidTransformError(const std::vector< std::vector<float> > refCoord, const std::vector< std::vector<float> > movCoord,
    const std::vector< std::vector<int> > RankInd,
    float * RTthis, float thresh2, const int topK, int *d_counts)
{
  int cnt = 0;
  std::vector<int> inliner;
  for (int i = 0; i < refCoord.size(); i++) {
    if (RankInd[i].size() > 0) {
      float min_err = std::numeric_limits<float>::max();
      int   min_err_ind = -1;
      for (int j = 0; j < comp_min(topK, RankInd[i].size()); j++) {
        float x1 = refCoord[i][0];
        float y1 = refCoord[i][1];
        float z1 = refCoord[i][2];

        float x2 = movCoord[RankInd[i][j]][0];
        float y2 = movCoord[RankInd[i][j]][1];
        float z2 = movCoord[RankInd[i][j]][2];

        float xt = RTthis[0] * x2 + RTthis[1] * y2 + RTthis[2] * z2 + RTthis[3];
        float yt = RTthis[4] * x2 + RTthis[5] * y2 + RTthis[6] * z2 + RTthis[7];
        float zt = RTthis[8] * x2 + RTthis[9] * y2 + RTthis[10] * z2 + RTthis[11];

        float err = (xt - x1) * (xt - x1) + (yt - y1) * (yt - y1) + (zt - z1) * (zt - z1);

        if (err < min_err) {
          min_err = err;
          min_err_ind = j;
        }


      }
      if (min_err < thresh2) {
        inliner.push_back(i);
        inliner.push_back(min_err_ind);
        cnt ++;
      }
    }
  }
  *d_counts = cnt;
  return inliner;

}

int ransacfitRt(const std::vector< std::vector<float> > refCoord, const std::vector< std::vector<float> > movCoord,
                const std::vector< std::vector<int> > RankInd, const int topK,
                const int numLoops, const float thresh, float* rigidtransform, bool is_verbose)
{
  //RankInd for every refCoord point top K matches RankInd.size()==refNum
  int refNum = refCoord.size();
  int movNum = movCoord.size();

  assert(RankInd.size() == refNum);

  int h_count = -1;
  float thresh2 = thresh * thresh;
  float h_RT[12];

  float x_in[9];
  float y_in[9];
  int sample[3];
  int maxCount = -1;
  std::vector<int> bestinlier;
  // int conv_num_loops = numLoops;
  for (int iter = 0; iter < numLoops; ++iter) {
    // Draw three points from refCoord
    int p1 = (int) floor(gen_random_float(0, refNum));
    int p2 = (int) floor(gen_random_float(0, refNum));
    int p3 = (int) floor(gen_random_float(0, refNum));
    // make sure they are different and has match in the movCoord
    while (RankInd[p1].size() == 0) p1 = ((int) floor(gen_random_float(0, refNum)));
    while (p2 == p1 || RankInd[p2].size() == 0) p2 = ((int) floor(gen_random_float(0, refNum)));
    while (p3 == p1 || p3 == p2 || RankInd[p3].size() == 0) p3 = ((int) floor(gen_random_float(0, refNum)));
    sample[0] = p1;
    sample[1] = p2;
    sample[2] = p3;



    // Find corresponding features in the movCoord
    int corr_sample[3];
    for (int i = 0; i < 3; i++) {
      int tmp_rand_ind = (int) floor(gen_random_float(0, comp_min(RankInd[sample[i]].size(), topK)));
      corr_sample[i] = RankInd[sample[i]][tmp_rand_ind];
      //corr_sample[i] = RankInd[sample[i]][0];
    }


    // Estimate the transform from the samples to their corresponding points
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        x_in[j + i * 3] = refCoord[sample[i]][j];
        y_in[j + i * 3] = movCoord[corr_sample[i]][j];
      }
    }


    estimateRigidTransform(x_in, y_in, 3, h_RT);  //x_in = RT*y_in
    // Tranform the data and compute the error
    std::vector<int> inlier = TestRigidTransformError(refCoord, movCoord, RankInd, h_RT, thresh2, topK, &h_count);

    if (h_count > maxCount) {
      maxCount = h_count;
      bestinlier = inlier;
      for (int i = 0; i < 12; i++) {
        rigidtransform[i] = h_RT[i];
      }

      if (is_verbose)
        printf("RANSAC iteration: %d/%d, inliers: %d\n", iter, numLoops, maxCount);
      
      // Update estimate of N, the number of trials to ensure we pick,
      // with probability p, a data set with no outliers.
      // float p = 0.9999f;
      // float frac_inliers =  (float) maxCount / (float) refCoord.size();
      // float pNoOutliers = 1 - frac_inliers * frac_inliers * frac_inliers;
      // pNoOutliers = comp_max(std::numeric_limits<float>::epsilon(), pNoOutliers);  // Avoid division by - Inf
      // pNoOutliers = comp_min(1 - std::numeric_limits<float>::epsilon(), pNoOutliers); // Avoid division by 0.
      // conv_num_loops = (int) std::floor(std::log(1 - p) / std::log(pNoOutliers));
      // conv_num_loops = comp_min(conv_num_loops, 100000); // at least try 10,000 times
    }

    // if (iter > conv_num_loops)
    //   break;

  }

  if (is_verbose)
    for (int jj = 0; jj < 3; jj++) {
      printf("%f,%f,%f,%f\n", rigidtransform[0 + jj * 4], rigidtransform[1 + 4 * jj], rigidtransform[2 + 4 * jj], rigidtransform[3 + 4 * jj]);
    }

  // reestimate RT based on inliners
  float * refCoord_inlier = new float[maxCount * 3];
  float * movCoord_inlier = new float[maxCount * 3];
  for (int i = 0; i < maxCount; ++i) {
    for (int j = 0; j < 3; j++) {
      refCoord_inlier[3 * i + j] = refCoord[bestinlier[2 * i]][j];
      int tmp_idx = RankInd[bestinlier[2 * i]][bestinlier[2 * i + 1]];
      movCoord_inlier[3 * i + j] = movCoord[tmp_idx][j];
    }
  }
  if (is_verbose) {
    printf("========================================================================================================================\n");
    printf("Total # of inliers: %d \n", maxCount);
  }

  estimateRigidTransform(refCoord_inlier, movCoord_inlier, maxCount, rigidtransform);
  delete[] refCoord_inlier;
  delete[] movCoord_inlier;

  if (is_verbose) {
    printf("========================================================================================================================\n");
    for (int jj = 0; jj < 3; jj++) {
      printf("%f,%f,%f,%f\n", rigidtransform[0 + jj * 4], rigidtransform[1 + 4 * jj], rigidtransform[2 + 4 * jj], rigidtransform[3 + 4 * jj]);
    }
  }
  return maxCount;
}

// int main(){
//   //float x_in[9] = {0,0,0,0,1,0,0,2,0};
//   //float y_in[9] = {0,0,0,0,2,0,0,3,0};
//   float x_in[9] = {0,0,0,0,1,0,0,2,0};
//   float y_in[9] = {-0.3,0.5,0,3,3,0,3,2,0};
//   float T[12];
//   estimateRigidTransform(x_in, y_in,3, T);
//   return 0;
// }


