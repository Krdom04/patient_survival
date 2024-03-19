#include <stdio.h>
#include <stdlib.h>
#include <math.h>

char m = 7, n = 2;

double sigmoid_j(double); //Converts log-odds to probability
double log_odds_pred(double *theta, double *b, float x[2][7], int ex_number); //Gives log-odds prediction for a training example
double cost(double *theta, double *b, float x[2][7], char y[]); //Calculates cost of parameters
void gradient_i(double *theta, double *b, float x[2][7], char y[], double *dj_dw, double *dj_db); //Calculates gradient of coefficients (dj_dw) and intercept (dj_db)
void gradient_desc(double *theta, double *b, float x[2][7], char y[], double alpha, unsigned int iters); //Calculate optimum parameters

int main(){

    //Initilize set
    m = 7;
    n = 2;
    float x[2][7] = {{16, 23, 12, 25, 30, 45, 15}, {0.3,1,0.6,0.8,0.5,0.3,1}};
    char y[] = {0,1,0,1,1,1,0};

    double theta_init[] = {0, 0};
    double b_init = 0;
    double alpha = 0.001;
    unsigned int iters = 300;
    gradient_desc(theta_init, &b_init,x,y,alpha,iters);
    printf("Coefficients: %lf, %lf, y-intercept: %lf\n", theta_init[0], theta_init[1], b_init);
    printf("Final cost is: %lf\n", cost(theta_init,&b_init,x,y));

    return 0;
}

void gradient_desc(double *theta, double *b, float x[2][7], char y[], double alpha, unsigned int iters){
    double dj_db = 0;
    double dj_dw[] = {0,0};

    for(int it=0;it<iters;it++){
        gradient_i(theta,b,x,y,dj_dw,&dj_db); //calculate gradient for each parameter
        *b -= alpha*dj_db;
        for(int i=0;i<n;i++){
            gradient_i(theta,b,x,y,dj_dw,&dj_db); //calculate gradient for each parameter
            *(theta+i) -= alpha*dj_dw[i];
        }
        if(it%30 == 0 || it==iters-1){
            printf("Iteration: %d cost: %lf\n", it, cost(theta,b,x,y));
        }
    }

}

void gradient_i(double *theta, double *b, float x[2][7], char y[], double *dj_dw, double *dj_db){
    double y_hat_j; //prediction/hypothesis
    double error;
    for(int j=0;j<m;j++){
        y_hat_j = sigmoid_j(log_odds_pred(theta,b,x,j));
        error = y_hat_j - y[j];
        *dj_db += error;
        for(int i=0;i<n;i++){
            *(dj_dw+i) += error*x[i][j];
        }
    }
    *dj_db /= m;
    for(int l=0;l<n;l++){
        *(dj_dw+l) /= m;
    }
}

double sigmoid_j(double z_j){
    return 1/(1+exp(-z_j));
}

double log_odds_pred(double *theta, double *b, float x[2][7], int ex_number){
    double result = 0;
    for(int i=0; i<n;i++){
        result += (*(theta+i)) * (x[i][ex_number]);
    }
    result += *b;
    return result;
}

double cost(double *theta, double *b, float x[2][7], char y[]){
    double cost = 0;
    double y_hat_j;
    for(int j=0;j<m;j++){
        y_hat_j = sigmoid_j(log_odds_pred(theta,b,x,j));

        cost += y[j]*log(y_hat_j);
        cost += (1-y[j])*log(1-y_hat_j);
    }
    cost /= m;
    cost *= -1;
    return cost;
}

