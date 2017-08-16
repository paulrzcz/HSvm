module Data.SVM where

-- TODO limitare l'export
-- TODO verificare l'import

import Control.Arrow ((***))
import Control.Monad (when, liftM)
import Control.Exception 
import Data.IntMap (IntMap, toList)
import qualified Data.IntMap as M
import Foreign.Storable (poke, peek)
import Foreign.Marshal.Alloc (malloc, alloca, free)
import Foreign.Marshal.Array
import Foreign.ForeignPtr 
import Foreign.Ptr (Ptr, nullPtr)
import Foreign.C.String 
import System.IO.Unsafe
import qualified Data.SVM.Raw as R
import Data.SVM.Raw (CSvmModel, CSvmProblem(..), CSvmNode(..), CSvmParameter,
                     c_svm_train, c_svm_cross_validation,
                     c_svm_destroy_model, c_svm_check_parameter,
                     c_svm_load_model, c_svm_save_model, c_svm_predict,
                     c_clone_model_support_vectors, defaultCParam)

type Vector = IntMap Double
type Problem = [(Double, Vector)]
newtype Model = Model (ForeignPtr CSvmModel)

data KernelType = Linear 
                | RBF     { gamma :: Double }
                | Sigmoid { gamma :: Double, coef0 :: Double }
                | Poly    { gamma :: Double, coef0 :: Double, degree :: Int}

data Algorithm = CSvc  { c :: Double }
               | NuSvc { nu :: Double }
               | NuSvr { nu :: Double, c :: Double }
               | EpsilonSvr { epsilon :: Double, c :: Double }
               | OneClassSvm { nu :: Double }

data ExtraParam = ExtraParam {cacheSize :: Double, 
                              shrinking :: Int, 
                              probability :: Int}

defaultExtra = ExtraParam {cacheSize = 100, shrinking = 1, probability = 0}

mergeKernel :: KernelType -> CSvmParameter -> CSvmParameter
mergeKernel Linear p        = p { R.kernel_type = R.linear }
mergeKernel (RBF g) p       = p { R.kernel_type = R.rbf,
                                  R.gamma = realToFrac g }
mergeKernel (Sigmoid g c) p = p { R.kernel_type = R.sigmoid,
                                  R.gamma = realToFrac g, 
                                  R.coef0 = realToFrac c }
mergeKernel (Poly g c d) p  = p { R.kernel_type = R.poly, 
                                  R.gamma = realToFrac g, 
                                  R.coef0 = realToFrac c, 
                                  R.degree = fromIntegral d}

mergeAlgo :: Algorithm -> CSvmParameter -> CSvmParameter
mergeAlgo (CSvc c) p         = p { R.svm_type = R.cSvc,  
                                   R.c = realToFrac c }
mergeAlgo (NuSvc nu) p       = p { R.svm_type = R.nuSvc, 
                                   R.nu = realToFrac nu }
mergeAlgo (NuSvr nu c) p     = p { R.svm_type = R.nuSvr, 
                                   R.nu = realToFrac nu, 
                                   R.c = realToFrac c }
mergeAlgo (EpsilonSvr e c) p = p { R.svm_type = R.epsilonSvr, 
                                   R.eps = realToFrac e, 
                                   R.c = realToFrac c }

mergeExtra (ExtraParam c s pr) p = p { R.cache_size = realToFrac c,
                                       R.shrinking = fromIntegral s,
                                       R.probability = fromIntegral pr }

-------------------------------------------------------------------------------

newCSvmNodeArray :: Vector -> IO (Ptr CSvmNode)
newCSvmNodeArray v = newArray (convertVector v ++ [CSvmNode (-1) 0])
            where convertVector :: Vector -> [CSvmNode]
                  convertVector = map convertNode . toList . M.filter (/= 0)
                  convertNode = uncurry CSvmNode . (fromIntegral *** realToFrac)

newCSvmProblem :: Problem -> IO (Ptr CSvmProblem)
newCSvmProblem lvs = do nodePtrList <- mapM newCSvmNodeArray $ map snd lvs
                        nodePtrPtr  <- newArray nodePtrList
                        labelPtr <- newArray . map realToFrac $ map fst lvs
                        let l = fromIntegral . length $ lvs
                        ptr <- malloc
                        poke ptr $ CSvmProblem l labelPtr nodePtrPtr
                        return ptr

freeCSVmProblem :: Ptr CSvmProblem -> IO ()
freeCSVmProblem ptr = do prob <- peek ptr
                         free $ y prob
                         vecList <- peekArray (fromIntegral $ l prob) (x prob)
                         mapM_ free vecList 
                         free $ x prob
                         free ptr

withProblem :: Problem -> (Ptr CSvmProblem -> IO a) -> IO a
withProblem prob = bracket (newCSvmProblem prob) freeCSVmProblem 

---

withParam :: ExtraParam 
             -> Algorithm 
             -> KernelType 
             -> (Ptr CSvmParameter -> IO a) 
             -> IO a
withParam extra algo kern f = 
    let merge = mergeAlgo algo . mergeKernel kern . mergeExtra extra 
        param = merge defaultCParam
    in alloca $ \paramPtr -> poke paramPtr param >> f paramPtr

checkParam :: Ptr CSvmProblem -> Ptr CSvmParameter -> IO ()
checkParam probPtr paramPtr = do
    let errStr = c_svm_check_parameter probPtr paramPtr
    when (errStr /= nullPtr) $ peekCString errStr >>= error . ("svm: "++)

--

train' :: ExtraParam -> Algorithm -> KernelType -> Problem -> IO (Model)
train' extra algo kern prob = 
    withProblem prob $ \probPtr ->
    withParam extra algo kern $ \paramPtr -> do
        checkParam probPtr paramPtr
        -- rereadParam <- peek paramPtr
        -- print rereadParam
        modelPtr <- c_svm_train probPtr paramPtr
        c_clone_model_support_vectors modelPtr
        modelForeignPtr <- newForeignPtr c_svm_destroy_model modelPtr
        return $ Model modelForeignPtr


-- | The 'train' function allows training a 'Model' starting from a 'Problem'
-- by specifying an 'Algorithm' and a 'KernelType'
train :: Algorithm -> KernelType -> Problem -> IO (Model)
train = train' defaultExtra

crossValidate' :: ExtraParam 
                  -> Algorithm 
                  -> KernelType 
                  -> Problem 
                  -> Int 
                  -> IO [Double]
crossValidate' extra algo kern prob nFold =
    withProblem prob $ \probPtr ->
    withParam extra algo kern $ \paramPtr -> do
        probLen <- (fromIntegral . R.l) `liftM` peek probPtr
        allocaArray probLen $ \targetPtr -> do -- (length prob is inefficient)
            checkParam probPtr paramPtr
            let c_nFold = fromIntegral nFold
            c_svm_cross_validation probPtr paramPtr c_nFold targetPtr
            map realToFrac `liftM` peekArray probLen targetPtr

crossValidate = crossValidate' defaultExtra

-----------------------------------------------------------------------

saveModel :: Model -> FilePath -> IO ()
saveModel (Model modelForeignPtr) path = 
    withForeignPtr modelForeignPtr $ \modelPtr -> do
        pathString <- newCString path
        ret <- c_svm_save_model pathString modelPtr
        when (ret /= 0) $ error "svm: error saving the model"

loadModel :: FilePath -> IO (Model)
loadModel path = do
    modelPtr <- c_svm_load_model =<< newCString path 
    Model `liftM` newForeignPtr c_svm_destroy_model modelPtr

---
predict :: Model -> Vector -> Double
predict (Model modelForeignPtr) vector = unsafePerformIO action
    where action :: IO Double
          action = withForeignPtr modelForeignPtr $ \modelPtr -> 
                   bracket (newCSvmNodeArray vector) free $ \vectorPtr ->
                        return . realToFrac . c_svm_predict modelPtr $ vectorPtr
        
