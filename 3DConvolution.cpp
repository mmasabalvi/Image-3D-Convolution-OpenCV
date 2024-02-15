


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// storing 3D image in 3d array
void store3DImage(Mat img, int*** arr)
{
	int rows = img.size().height;                                                                               //image pixel intensity values stored in the matrix here
	int cols = img.size().width;                                                                                //cols and rows set 

	
	for (int i = 0;i < rows;i++) 
	{
		for (int j = 0;j < cols;j++)                                                                        // each intensity values stored in the array
        {	
            arr[i][j][0] = static_cast<int>(img.at<uchar>(i, j));                                                   //at<uchar>() inbuilt function of OpenCv
		}                                                                                                   //we can access pixel values at the cordinates given
	}                                                                                                           //in this case i and j
                                                                                                                    //so each cordinate read, and stored in arr
	/*for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << arr[i][j] << " ";                                                                   //testing purpose
		}

		cout << endl;
	}*/
}

void print3DImage(int*** arr, int rows, int cols) 
{
	                                                                                                          //new Mat object to store the image through array given
	Mat image(rows, cols, CV_8UC1);                                                                           // CV_8UC1 represents grayscale image

    for (int i = 0; i < rows; i++)                                                                                //opposite of store3Dimage is happening
    {                                                                                                             //pixel intensity value stored in the matrix is being stored at the speicifc cordinates
		for (int j = 0; j < cols; j++)
        {
			image.at<uchar>(i, j) = static_cast<uchar>(arr[i][j][0]);                                 //at unchar() accesses the specific pixel
		}                                                                                                 //it reads values from the array already stored
	}                                                                                                         // and converts it to unchar to store it there

	
	imshow("Image", image);                                                                                    //im show prints the image
	waitKey(0);                                                                                               //when key pressed, the window will close here
	//destroyAllWindows(); 
}

//Apply 3D Convolution for Blurring
void apply3DConvolution(int*** arr, int rows, int cols)
{
    // Define a simple averaging kernel
    float kernel[3][3] = {{1.0 / 9, 1.0 / 9, 1.0 / 9}, {1.0 / 9, 1.0 / 9, 1.0 / 9}, {1.0 / 9, 1.0 / 9, 1.0 / 9}};
													      //averaging kernel is used (box filter)
                                                                                                              //basically all its values have the same weight factor that is 1.0/9.
                                                                                                              //so it multiplies with the neighbiring array and can find the average of the values to store in the middle pixel of coresponding array
    
    int*** blurr = new int** [rows];                                                                          //array created of the same size to store blur image

    for (int i = 0; i < rows; i++)                                                                            //dynamically declared using rows and cols
    {
        blurr[i] = new int* [cols];

        for (int j = 0; j < cols; j++)
        {
            blurr[i][j] = new int[1];                                                                        //grayscale image in 3d has 3rd dimension 1
                                                                                                             //1 stands for grayscale
        }
    }


    for (int i = 0; i < rows; i++)                                                                            //loop i for image arr rows
    {
        for (int j = 0; j < cols; j++)                                                                        //loop j for image arr cols
        {
            float sum = 0;                                                                                    //variable sum for summing the matrix and kernel

            for (int k = 0; k < 3; k++)                                                                       //loop k for kernel rows
            {
                for (int l = 0; l < 3; l++)                                                                   //loop l for kernel cols
                {
                    int neighborX = i + k - 1;                                                                //neighboring pixels are calculated
                    int neighborY = j + l - 1;                                                                //for X, i is the center pixel's row
                                                                                                              //the kernel's first row will be one row above i 
                                                                                                              //so 1 subtracted from each entry
                    
                    if (neighborX < 0 || neighborX >= rows)                                                   //zero padding done here
                    {                                                                                         //if neighboring pixel out of range, then sum unchanged
                        sum += 0;                                                                             //used for borders
                    }                                                                                         //so that the kernel matrix multiplication does not go out of range

                    else if (neighborY < 0 || neighborY >= cols)                                              //same principle for y cordinates
                    {
                        sum += 0;
                    }

                    else                                                                                      //neighboring pixels multiplied with coresponding kernel values
                    {                                                                                         //works only if neighboring pixels are not out of range
                        sum += kernel[k][l] * arr[neighborX][neighborY][0];                                   //values added in sum
                    }                                                                                         //indirectly avergaged as each kernel value is 1/9 and is avergaed through it
                }                                                                                             //same weight aplplied for all entries
            }

            blurr[i][j][0] = int(sum);                                                                         //sum then stored at the center pixel
        }   
    }

  
    print3DImage(blurr, rows, cols);                                                                            //image printed

  
    Mat BlurredImage(rows, cols, CV_8UC1);                                                                      //to save file, new Mat object made

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            BlurredImage.at<uchar>(i, j) = uchar(blurr[i][j][0]);                                               //blurr matrix values stored in new blur images specific cordinates
        }
    }

    imwrite("BlurredImage.png", BlurredImage);                                                                  //new image named blurred image saved and made through imwrite

        
}



// Apply 3D Convolution for Edge Detection:
void apply3DConvolutionEdgeDetection(int*** arr, int rows, int cols)                                             //used to identify boundaries and within an image by checking its pixel intensity
{
    float kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};                                                  //sobel kernel used for edge detetction
                                                                                                                 //2 matrices, for horizontal and vertical edges both
    float kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};                                                  //closer peixels have more effect so higher weight
                                                                                                                 //matrices are symmetric, range from -2 to +2, so change in intensity evident 
                                                                                                                 //final image is gradient image, boundaries of original image recognised using change in intensity

    int*** edge = new int** [rows];                                                                              //array created of the same size to store edgedetetcted image

    for (int i = 0; i < rows; i++)                                                                               //dynamically declared using rows and cols
    {
        edge[i] = new int* [cols];

        for (int j = 0; j < cols; j++)
        {
            edge[i][j] = new int[1];                                                                             //grayscale image in 3d has 3rd dimension 1
                                                                                                                 //1 stands for grayscale
        }
    }


    for (int i = 0; i < rows; i++)                                                                               //loop i for image arr rows
    {
        for (int j = 0; j < cols; j++)                                                                           //loop j for image arr cols
        {
            float sumX = 0;                                                                                      //variable sum for summing the matrix and kernel
            float sumY = 0;                                                                                      //variable sum for summing the matrix and kernel

            for (int k = 0; k < 3; k++)                                                                          //loop k for kernel rows
            {   
                for (int l = 0; l < 3; l++)                                                                      //loop l for kernel cols
                {
                    int neighborX = i + k - 1;                                                                   //neighboring pixels are calculated
                    int neighborY = j + l - 1;                                                                   //for X, i is the center pixel's row
                                                                                                                 //the kernel's first row will be one row above i 
                                                                                                                 //so 1 subtracted from each entry
                    
                    if (neighborX < 0 || neighborX >= rows)                                                     //zero padding done here
                    {                                                                                           //if neighboring pixel out of range, then iteration skipped
                        sumX += 0;                                                                               //used for borders
                    }                                                                                           //so that the kernel matrix multiplication does not go out of range

                    else if (neighborY < 0 || neighborY >= cols)                                                //same principle for y cordinates
                    {
                        sumY += 0;
                    }

                    else
                    {
                        sumX += kernelX[k][l] * arr[neighborX][neighborY][0];                                    //neighboring pixels multiplied with coresponding kernel values
                        sumY += kernelY[k][l] * arr[neighborX][neighborY][0];                                    //for both x and y axis
                    }
                }
            }

                                                                                                                 //both gradients are perpendicular to each other
            float mag = sqrt((sumX * sumX) + (sumY * sumY));                                                     //so magnitude calculated 
                                                                                                                 //mag represents strength of edge at pixel
           
            edge[i][j][0] = static_cast<int>(mag);                                                               //magnitude stored in array
        }
    }

    print3DImage(edge, rows, cols);                                                                              //image printed
  

    Mat EdgeImage(rows, cols, CV_8UC1);                                                                          //to save file, new Mat object made

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            EdgeImage.at<uchar>(i, j) = uchar(edge[i][j][0]);                                                    //edge matrix values stored in new edge detection images specific cordinates
        }
    }

    imwrite("EdgeDetectedImage.png", EdgeImage);                                                              //new image named EdgeDetected image saved and made through imwrite



}

// Apply 3D Convolution for Feature Extraction:
void apply3DConvolutionFeatureExtraction(int*** arr, int rows, int cols)                                      //used to identify patterns within image
{                                                                                           
    
    float kernel[3][3] = { {1, 0, 1}, {0, 1, 0}, {1, 0, 1} };                                                 //just used a random kernel, experimented with values

    
    int*** extraction = new int** [rows];                                                                     //array created of the same size to store image

    for (int i = 0; i < rows; i++)                                                                            //dynamically declared using rows and cols
    {
        extraction[i] = new int* [cols];

        for (int j = 0; j < cols; j++)
        {
            extraction[i][j] = new int[1];                                                                    //grayscale image in 3d has 3rd dimension 1
                                                                                                              //1 stands for grayscale
        }
    }


    for (int i = 0; i < rows; i++)                                                                            //loop i for image arr rows
    {
        for (int j = 0; j < cols; j++)                                                                        //loop j for image arr cols
        {
            float sum = 0;                                                                                    //variable sum for summing the matrix and kernel

            for (int k = 0; k < 3; k++)                                                                       //loop k for kernel rows
            {
                for (int l = 0; l < 3; l++)                                                                   //loop l for kernel cols
                {
                    int neighborX = i + k - 1;                                                                //neighboring pixels are calculated
                    int neighborY = j + l - 1;                                                                //for X, i is the center pixel's row
                                                                                                              //the kernel's first row will be one row above i 
                                                                                                              //so 1 subtracted from each entry

                    if (neighborX < 0 || neighborX >= rows)                                                   //zero padding done here
                    {                                                                                         //if neighboring pixel out of range, then sum unchanged
                        sum += 0;                                                                             //used for borders
                    }                                                                                         //so that the kernel matrix multiplication does not go out of range

                    else if (neighborY < 0 || neighborY >= cols)                                              //same principle for y cordinates
                    {
                        sum += 0;
                    }

                    else                                                                                      //neighboring pixels multiplied with coresponding kernel values
                    {                                                                                         //works only if neighboring pixels are not out of range
                        sum += kernel[k][l] * arr[neighborX][neighborY][0];                                   //values added in sum
                    }                                                                                         
                }                                                                                             
            }

            extraction[i][j][0] = int(sum);                                                                   //sum then stored at the center pixel
        }
    }


    print3DImage(extraction, rows, cols);                                                                     //image printed

    

    Mat ExtractionImage(rows, cols, CV_8UC1);                                                                 //to save file, new Mat object made

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            ExtractionImage.at<uchar>(i, j) = uchar(extraction[i][j][0]);                                     //extraction matrix values stored in new extraction images specific cordinates
        }
    }

    imwrite("FeatureExtractedImage.png", ExtractionImage);                                                    //new image named FeatureExtractedImage image saved and made through imwrite



}

int main() 
{

	string path = "image-1.tiff";                   						      //storing the path of the immage
	Mat image = imread(path, IMREAD_GRAYSCALE);                                                           //creating an image and reading it as grayscale

	//imshow("Image", image);
	//cv::waitKey(0); 
	//cv::destroyAllWindows(); 

	int rows = image.size().height;                                                                        //image rows set
	int cols = image.size().width;                                                                         //image cols set
	int height = 1;                                                                                        //grayscale image in 3d has 3rd dimension 1
                                                                                                               //1 stands for grayscale
	int*** arr = new int** [rows];                                                                         //3d array declaration

	for (int i = 0; i < rows; i++)                                                                         //Array dynamically declared
	{
		arr[i] = new int* [cols];

		for (int j = 0; j < cols; j++)
		{
			arr[i][j] = new int[height];
		}
	}

	store3DImage(image, arr);                                                                               //store image function called

	print3DImage(arr, rows, cols);                                                                          //original image printed

    apply3DConvolution(arr, rows, cols);                                                                        //Blurred image

    apply3DConvolutionEdgeDetection(arr, rows, cols);                                                           //Edge Detection Image

    apply3DConvolutionFeatureExtraction(arr, rows, cols);                                                       //Feature Extraction Image
	
}
