using System;
using System.Drawing;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Management;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using uEye;

namespace SafetyModule
{
    public enum ProcessState
    {
        Init,
        Safe,
        NotSafe,
        Paused,
        Terminated
    }
    public enum Command
    {
        Begin,
        End,
        Pause,
        Resume,
        Exit
    }

    public class SafetyModule
    {
        private static int instances = 0;
        private Image<Gray, byte> imgPreProcessed = null;
        private Image<Ycc, byte> imgPre = null;
        private Image<Gray, byte> imgGrayProcessed = null;
        public Image<Bgr, byte> imgContourHull = null;
        private Image<Bgr, byte> imgSubstract = null;
        private Image<Bgr, byte> imgback = null;

        private bool isSafe = false;

        private Camera __uEyeCamera = null;

        Dictionary<int, Image<Bgr, byte>> imgbackbuffer = new Dictionary<int, Image<Bgr, byte>>();

        public Ycc YCrCb_min = new Ycc(0, 129, 76);         //Paramterwerte für den YCC Filter
        public Ycc YCrCb_max = new Ycc(255, 176, 144);

        public int refreshBackCounter = 0;
        public double prevLuminance = 0;

        int highest = 0;
        int lowest = 0;

        public bool IsSafe
        {
            get { return isSafe; }
        }

        /// <summary>
        /// Create an instance of the SafetyModule. Only one instance exists at a time.
        /// </summary>
        /// <param name="uEyeCamera">Object of the uEyeCamera</param>
        public SafetyModule(Camera uEyeCamera)
        {
            if (SafetyModule.instances > 0)
                return;

            __uEyeCamera = uEyeCamera;

            __uEyeCamera.EventFrame += OnFrameEvent;

            StartUnplugWatcher();

            CurrentState = ProcessState.Init;
            transitions = new Dictionary<StateTransition, ProcessState>
            {
                { new StateTransition(ProcessState.Init, Command.Exit), ProcessState.Terminated },
                { new StateTransition(ProcessState.Init, Command.Begin), ProcessState.NotSafe },
                { new StateTransition(ProcessState.NotSafe, Command.Begin), ProcessState.Safe },
                { new StateTransition(ProcessState.Safe, Command.Pause), ProcessState.Paused },
                { new StateTransition(ProcessState.Safe, Command.End), ProcessState.NotSafe },
                { new StateTransition(ProcessState.Paused, Command.End), ProcessState.NotSafe },
                { new StateTransition(ProcessState.Paused, Command.Resume), ProcessState.Safe }
            };
        }

        private void OnFrameEvent(object sender, EventArgs e)
        {
            Image<Bgr, byte> currentFrame;

            uEye.Defines.Status statusRet = 0;

            // Get last image memory
            statusRet = __uEyeCamera.Memory.GetLast(out int s32LastMemId);
            statusRet = __uEyeCamera.Memory.Lock(s32LastMemId);
            statusRet = __uEyeCamera.Memory.GetSize(s32LastMemId, out int s32Width, out int s32Height);

            statusRet = __uEyeCamera.Memory.ToBitmap(s32LastMemId, out Bitmap bmpMyBitmap);

            if (statusRet == uEye.Defines.Status.Success)
            {
                // clone bitmap
                Rectangle cloneRect = new Rectangle(0, 0, s32Width, s32Height);
                System.Drawing.Imaging.PixelFormat format = System.Drawing.Imaging.PixelFormat.Format32bppArgb;

                if (imgback == null)
                {
                    Thread.Sleep(1500);
                    imgback = new Image<Bgr, byte>(bmpMyBitmap.Clone(cloneRect, format));
                }

                currentFrame = new Image<Bgr, byte>(bmpMyBitmap.Clone(cloneRect, format));

                DetectHumanSkin(currentFrame, out isSafe);

                // unlock image buffer
                statusRet = __uEyeCamera.Memory.Unlock(s32LastMemId);

                // Prevent RAM Out of Memory Exception
                GC.Collect();
            }
        }

        private void StartUnplugWatcher()
        {
            WqlEventQuery removeQuery = new WqlEventQuery("SELECT * FROM __InstanceDeletionEvent WITHIN 2 WHERE TargetInstance ISA 'Win32_USBHub'");
            ManagementEventWatcher removeWatcher = new ManagementEventWatcher(removeQuery);
            removeWatcher.EventArrived += new EventArrivedEventHandler(DeviceRemovedEvent);
            removeWatcher.Start();
        }

        private void DeviceRemovedEvent(object sender, EventArrivedEventArgs e)
        {
            ManagementBaseObject instance = (ManagementBaseObject)e.NewEvent["TargetInstance"];

            if (instance.Properties["Caption"].Value.ToString() == "uEye UI-164xLE Series")
            {
                isSafe = false;

                MessageBox.Show("Kamera wurde entfernt! Verbinden Sie die Kamera erneut.");
            }
        }

        private class StateTransition
        {
            readonly ProcessState CurrentState;
            readonly Command Command;

            public StateTransition(ProcessState currentState, Command command)
            {
                CurrentState = currentState;
                Command = command;
            }

            public override int GetHashCode()
            {
                return 17 + 31 * CurrentState.GetHashCode() + 31 * Command.GetHashCode();
            }

            public override bool Equals(object obj)
            {
                StateTransition other = obj as StateTransition;
                return other != null && this.CurrentState == other.CurrentState && this.Command == other.Command;
            }
        }

        Dictionary<StateTransition, ProcessState> transitions;
        public ProcessState CurrentState { get; private set; }

        public ProcessState GetNext(Command command)
        {
            StateTransition transition = new StateTransition(CurrentState, command);
            ProcessState nextState;
            if (!transitions.TryGetValue(transition, out nextState))
                throw new Exception("Invalid transition: " + CurrentState + " -> " + command);
            return nextState;
        }

        public ProcessState MoveNext(Command command)
        {
            CurrentState = GetNext(command);
            return CurrentState;
        }

        private Image<Bgr, byte> HaarClassifier(Image<Bgr, byte> imgCurrentFrame, out bool humanStrucDetected)
        {
            humanStrucDetected = false;
            Image<Gray, byte> imgGray = imgCurrentFrame.Convert<Gray, Byte>();

            Rectangle[] rectangles = { };
            for (int rotations = 0; rotations < 3; rotations++)
            {
                //rectangles = hand_cascade.DetectMultiScale(imgGray, 1.1d, 2);
                if (0 != rectangles.Length)
                    break;

                imgGray.Rotate(90.0d, new Gray(0));
            }

            for (int countRect = 0; countRect < rectangles.Length; countRect++)
            {
                humanStrucDetected = true;

                imgCurrentFrame.Draw(rectangles[countRect], new Bgr(200, 125, 75), 2);
            }

            return imgCurrentFrame;
        }
        /// <summary>
        /// Search for biggest human structure inside a image, mark it's contour and draw a convex hull around it.
        /// </summary>
        /// <param name="imgCurrentFrame">Image to process</param>
        /// <param name="humanStrucDetected">True: Human structure was found in the image, False: No human structure was found</param>
        /// <returns>Processed image with the marked contour and the convex hull around the biggest human structure, if any human structure was found.</returns>
        private void DetectHumanSkin(Image<Bgr, byte> imgCurrentFrame, out bool humanStrucDetected)
        {
            //  imgContourHull = HaarClassifier(imgCurrentFrame, out humanStrucDetected);
            // filter the image in terms of color and improve countour boundaries

            if (CurrentState == ProcessState.Init)
            {
                RefreshBackground(imgCurrentFrame, 5.0);
                MoveNext(Command.Begin);
            }

            imgGrayProcessed = ImgPreProcessing(imgCurrentFrame);

            // find a contour which represents a hand and draw a hull
            //imgContourHull = ExtractContourAndHull(imgGrayProcessed, imgCurrentFrame, out humanStrucDetected);
            if (CurrentState == ProcessState.Safe || CurrentState == ProcessState.NotSafe)
            {
                imgContourHull = ExtractContourAndHull(imgGrayProcessed, imgCurrentFrame, out humanStrucDetected);
            }
            else
            {
                humanStrucDetected = true;
            }
            RefreshBackground(imgCurrentFrame, 20.0);
        }

        private static byte SaturateCast(double value)
        {
            var rounded = Math.Round(value, 0);

            if (rounded < byte.MinValue)
            {
                return byte.MinValue;
            }

            if (rounded > byte.MaxValue)
            {
                return byte.MaxValue;
            }

            return (byte)rounded;
        }

        private Image<Gray, Byte> SkinFilter(Image<Ycc, Byte> imgCurrentFrame)
        {
            Image<Gray, Byte>[] channels = imgCurrentFrame.Split();
            Image<Gray, Byte> imgFiltered = new Image<Gray, byte>(imgCurrentFrame.Width, imgCurrentFrame.Height);

            channels[1].InRange(new Gray(132.0), new Gray(173.0));
            channels[2].InRange(new Gray(76.0), new Gray(126.0));

            imgFiltered = channels[1].And(channels[2]);

            //imgFiltered = imgFiltered.ThresholdBinary

            imgPreProcessed = imgFiltered;

            return imgFiltered;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="Img"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        private Image<Gray, byte> DetectSkin(Image<Bgr, byte> Img, IColor min, IColor max)
        {
            Image<Ycc, Byte> currentYCrCbFrame = Img.Convert<Ycc, Byte>();
            Image<Gray, byte> skin = new Image<Gray, byte>(Img.Width, Img.Height);
            skin = currentYCrCbFrame.InRange((Ycc)min, (Ycc)max);
            Mat rect_12 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Ellipse, new System.Drawing.Size(5, 5), new Point(-1, -1));

            CvInvoke.Erode(skin, skin, rect_12, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(1.0));
            Mat rect_6 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Ellipse, new System.Drawing.Size(5, 5), new Point(-1, -1));

            CvInvoke.Dilate(skin, skin, rect_6, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(1.0));

            return skin;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="imgCurrentFrame"></param>
        /// <returns></returns>
        private Image<Ycc, Byte> EqualizeImgLuminance(Image<Bgr, Byte> imgCurrentFrame)
        {
            Image<Ycc, Byte> imgYcc = imgCurrentFrame.Convert<Ycc, Byte>();

            Image<Gray, Byte>[] channels;
            channels = imgYcc.Split();

            // Dunkelstufe (Channel 3) anpassen
            channels[0]._GammaCorrect(0.9);

            Image<Ycc, Byte> imgEqualizedLuminance = imgYcc;
            VectorOfMat arrChannels = new VectorOfMat(channels[0].Mat, channels[1].Mat, channels[2].Mat);
            CvInvoke.Merge(arrChannels, imgEqualizedLuminance);

            return imgEqualizedLuminance;
        }

        /// <summary>
        /// Filter the given image by human skin color and improve the boundaries of the remaining structures.
        /// </summary>
        /// <param name="imgCurrentFrame">The image which should be progressed.</param>
        /// <returns>Gray Image which can contain a hand.</returns>
        private Image<Gray, byte> ImgPreProcessing(Image<Bgr, byte> imgCurrentFrame)
        {
            if (imgback == null)
            {
                return null;
            }

            Image<Bgr, byte> dif = imgCurrentFrame.AbsDiff(imgback);
            dif = dif.Not();


            Image<Gray, byte> imgOutput = dif.Convert<Gray, byte>().InRange(new Gray(245), new Gray(255));

            if (dif == null)
            {
                return null;
            }
            Image<Bgr, byte> temp = imgCurrentFrame.Clone();
            temp.SetValue(new Bgr(255, 255, 255), imgOutput);

            temp._EqualizeHist();

            // create gray image with the same size as image
            Image<Gray, byte> imgGray = new Image<Gray, byte>(imgCurrentFrame.Width, imgCurrentFrame.Height);

            imgGray = DetectSkin(temp, YCrCb_min, YCrCb_max);

            imgSubstract = temp;

            Mat kernel = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new System.Drawing.Size(5, 5), new Point(-1, -1));

            imgGray = imgGray.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Close, kernel, new Point(-1, -1), 6, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(1.0));

            return imgGray;
        }



        /// <summary>
        /// draw Contur of the Hand 
        /// </summary>
        /// <param name="imgGray"></param>
        /// <param name="imgCurrentFrame"></param>
        /// <param name="humanStrucDetected"></param>
        /// <returns></returns>
        private Image<Bgr, byte> ExtractContourAndHull(Image<Gray, byte> imgGray, Image<Bgr, byte> imgCurrentFrame, out bool humanStrucDetected)
        {
            if (CurrentState == ProcessState.Paused)
            {
                humanStrucDetected = false;
                return imgCurrentFrame;
            }

            // create variable where found contours can be stored
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();

            // convert Gray-Image to UMat
            UMat grayMat = imgGray.ToUMat();

            // search for contours in the Gray-Image
            CvInvoke.FindContours(grayMat, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            #region Find biggest contour
            VectorOfPoint biggestContour = null;

            double biggestArea = 0;
            double contourArea = 0;

            for (int contourNumber = 0; contourNumber < contours.Size; contourNumber++)
            {
                // get current contours area
                contourArea = CvInvoke.ContourArea(contours[contourNumber], false);

                // check if area is at least 8100.0 and if it's greater than the last contours area
                if (contourArea > biggestArea && contourArea > 1000.0)
                {
                    biggestArea = contourArea;
                    biggestContour = contours[contourNumber];
                }
            }
            #endregion

            // check if a contour was found
            if (biggestContour != null)
            {
                humanStrucDetected = true;

                #region Approx curve around the biggest contour and draw it
                VectorOfPoint currentContour = new VectorOfPoint();

                CvInvoke.ApproxPolyDP(biggestContour, currentContour, CvInvoke.ArcLength(biggestContour, true) * 0.0025, true);

                Point[] currentContourPoints = currentContour.ToArray();

                imgCurrentFrame.Draw(currentContourPoints, new Bgr(System.Drawing.Color.LimeGreen), 2);
                #endregion

                #region Find convex hull of the contour and draw it
                PointF[] hullPoints = CvInvoke.ConvexHull(Array.ConvertAll<Point, PointF>(currentContourPoints, new Converter<Point, PointF>(PointToPointF)), true);

                RotatedRect rotatedRect = CvInvoke.MinAreaRect(currentContour);

                PointF[] pointFs = rotatedRect.GetVertices();

                Point[] points = new Point[pointFs.Length];

                for (int i = 0; i < pointFs.Length; i++)
                    points[i] = new Point((int)pointFs[i].X, (int)pointFs[i].Y);

                imgCurrentFrame.DrawPolyline(Array.ConvertAll<PointF, Point>(hullPoints, Point.Round),
                    true, new Bgr(200, 125, 75), 2);

                imgCurrentFrame.Draw(new CircleF(new PointF(rotatedRect.Center.X, rotatedRect.Center.Y), 3),
                    new Bgr(200, 125, 75), 2);
                #endregion

                if (CurrentState == ProcessState.Safe)
                {
                    MoveNext(Command.End);
                }
            }
            else
            {
                humanStrucDetected = false;
                //RefreshBackground(imgCurrentFrame, 2.0);

                if (CurrentState == ProcessState.NotSafe)
                {
                    MoveNext(Command.Begin);
                }


            }
            //RefreshBackground(imgCurrentFrame, 2.0);

            return imgCurrentFrame;
        }


        private void RefreshBackground(Image<Bgr, byte> imgCurrentFrame, double refreshHysteresis)
        {
            Image<Gray, byte> imgGrayCurrent = imgCurrentFrame.Convert<Gray, byte>();


            Image<Hsv, byte> imgHsvCurrent = imgCurrentFrame.Convert<Hsv, byte>();          // new HSV image of the current frame to compare with the background 
            Image<Hsv, byte> imgHsvCurrentBackground = imgback.Convert<Hsv, byte>();        // new HSV image of the current background frame to compare with the current frame

            double VvalueCurrent = imgHsvCurrent.GetSum().Value / (imgCurrentFrame.Width * imgCurrentFrame.Height);   // get the luminance 
            double VvalueBackground = imgHsvCurrentBackground.GetSum().Value / (imgHsvCurrentBackground.Width * imgHsvCurrentBackground.Height);


            double brightness_imgGrayCurrent = imgGrayCurrent.GetAverage().Intensity;



            double LuminanceDiff = Math.Abs(VvalueCurrent - VvalueBackground);


            YCrCb_min = new Ycc(0, VvalueBackground * (6.0d / 90.0d) + 122.31, 76);

            //__SaftyInterface.InvokeEx(f => f.WriteLogInfo("LuminanceDiff = " + LuminanceDiff));

            // __SaftyInterface.InvokeEx(f => f.WriteLogInfo("Luminance = " + VvalueCurrent));

            //__SaftyInterface.InvokeEx(f => f.WriteLogInfo("CrMin = " + (VvalueBackground * (6.0d / 90.0d) + 122).ToString()));

            if (CurrentState == ProcessState.Safe)
            {
                if (imgbackbuffer.Count == 0)
                {
                    imgbackbuffer.Add(Convert.ToInt32(VvalueBackground), imgback);
                    highest = Convert.ToInt32(VvalueBackground);
                    lowest = Convert.ToInt32(VvalueBackground);
                }

                if (imgbackbuffer.Count < 200)
                {
                    if (!imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent)))
                    {
                        imgbackbuffer.Add(Convert.ToInt32(VvalueCurrent), imgCurrentFrame);

                        foreach (KeyValuePair<int, Image<Bgr, byte>> entry in imgbackbuffer)
                        {
                            if (highest < entry.Key)
                            {
                                highest = entry.Key;
                            }

                            if (lowest > entry.Key)
                            {
                                lowest = entry.Key;
                            }
                            // do something with entry.Value or entry.Key
                        }
                    }
                    else
                    {
                        imgbackbuffer[Convert.ToInt32(VvalueCurrent)] = imgCurrentFrame;
                    }

                }

            }
            else if (CurrentState == ProcessState.Init)
            {
                /*
                for (int i = 1; i <= 78; i++)
                {
                    Image<Bgr, byte> imgInput = new Image<Bgr, byte>(@"C:\DHBW\Studienarbeit\T3200\Software\SA2_SafetyModule\SafetyModule\Resources\back\back" + i + ".png");

                    imgHsvCurrent = imgInput.Convert<Hsv, byte>();

                    VvalueCurrent = imgHsvCurrent.GetSum().Value / (imgInput.Width * imgInput.Height);




                    if (!imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent)))
                    {
                        imgbackbuffer.Add(Convert.ToInt32(VvalueCurrent), imgInput);
                    }

                }
                */
                /*
                    for (int i = 1; i < 5; i++)
                    {
                        //continue;
                        Image<Gray, Byte>[] channels;
                        channels = imgHsvCurrent.Split();
                        channels[2]._GammaCorrect(1 + (i / 10));
                        //channels[2]._GammaCorrect(3);

                        Image<Hsv, byte> imgEqualizedLuminance = imgHsvCurrent;


                        VectorOfMat arrChannels = new VectorOfMat(channels[0].Mat, channels[1].Mat, channels[2].Mat);
                        CvInvoke.Merge(arrChannels, imgEqualizedLuminance);

                        Image<Bgr, byte> imgEqualized = imgEqualizedLuminance.Convert<Bgr, byte>();


                        imgHsvCurrent = imgEqualized.Convert<Hsv, byte>();

                        VvalueCurrent = imgHsvCurrent.GetSum().Value / (imgEqualized.Width * imgEqualized.Height);

                        //VvalueCurrent = channels[2].GetSum().value;

                        VvalueCurrent = VvalueCurrent - (i * 10);


                        //imgEqualized.Save("C:\\temp\\Bilder" + "\\" + "row_image" + i);



                        if (!imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent)))
                        {
                            imgbackbuffer.Add(Convert.ToInt32(VvalueCurrent), imgEqualized);
                        }

                }
                    */
            }



            if (LuminanceDiff > refreshHysteresis)
            {
                if (CurrentState == ProcessState.Safe)
                {
                    MoveNext(Command.Pause);
                }
                else if (CurrentState == ProcessState.NotSafe)
                {
                    for (int i = 0; i < 100; i++)
                    {
                        if (imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent) - i))
                        {
                            imgback = imgbackbuffer[Convert.ToInt32(VvalueCurrent) - i];
                            return;
                        }

                        if (imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent) + i))
                        {
                            imgback = imgbackbuffer[Convert.ToInt32(VvalueCurrent) + i];
                            return;
                        }
                    }
                }

                if (CurrentState == ProcessState.Paused)
                {
                    for (int i = 0; i < 100; i++)
                    {
                        if (imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent) - i))
                        {
                            imgback = imgbackbuffer[Convert.ToInt32(VvalueCurrent) - i];
                            return;
                        }

                        if (imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent) + i))
                        {
                            imgback = imgbackbuffer[Convert.ToInt32(VvalueCurrent) + i];
                            return;
                        }
                    }
                }
            }
            else if (CurrentState == ProcessState.Paused)
            {
                MoveNext(Command.Resume);
                refreshBackCounter++;
            }
            return;


            if (LuminanceDiff > refreshHysteresis)
            {
                if (CurrentState == ProcessState.Safe)
                {
                    MoveNext(Command.Pause);
                }


                if (Math.Abs(prevLuminance - LuminanceDiff) > 1)
                {
                    prevLuminance = LuminanceDiff;
                    //imgback = imgCurrentFrame;
                    if (VvalueCurrent > VvalueBackground)
                    {
                        //  imgback._GammaCorrect(3);

                        //imgback = imgback.Not();


                    }
                    //imgback = imgCurrentFrame;
                }
                else
                {
                    //imgback = imgCurrentFrame;



                    if (imgbackbuffer.ContainsKey(Convert.ToInt32(VvalueCurrent)))
                    {
                        imgback = imgbackbuffer[Convert.ToInt32(VvalueCurrent)];
                    }


                    MoveNext(Command.Resume);
                    refreshBackCounter++;
                    prevLuminance = 0;
                }

            }
            else
            {
                if (CurrentState == ProcessState.Paused)
                {
                    MoveNext(Command.Resume);
                }
            }

        }

        /// <summary>
        /// Converts a Point structure to a PointF structure
        /// </summary>
        /// <param name="pt"></param>
        /// <returns></returns>
        public static PointF PointToPointF(Point pt)
        {
            return new PointF(((float)pt.X), ((float)pt.Y));
        }
    }

}
