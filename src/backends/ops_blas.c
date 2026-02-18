#include "../../include/matrix.h"
#include <cblas.h>

Matrix mat_mul_blas(Matrix A, Matrix B) {
    Matrix C = create_matrix(A.rows, B.columns);

    
    cblas_sgemm(
        CblasRowMajor, 
        CblasNoTrans,  
        CblasNoTrans,  
        A.rows,        
        B.columns,     
        A.columns,     
        1.0f,          
        A.data,        
        A.columns,     
        B.data,        
        B.columns,     
        0.0f,          
        C.data,        
        C.columns      
    );

    return C;
}