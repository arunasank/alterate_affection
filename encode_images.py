import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import datetime
# URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

URL_FFHQ = 'https://drive.google.com/uc?id=1PrjiBIXQ-tEa3l3vAtBR7FLi-n9WnVB8' #personal 

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
"""
Find latent representation of reference images using perceptual loss
"""
def encode(src_dir,                 #Directory with images for encoding'
           generated_images_dir,    #Directory with images for encoding
           dlatent_dir,             #Directory for storing dlatent representations
           batch_size=1,            #Batch size for generator and perceptual model
           image_size= 256,         #Size of images for perceptual model
           lr=1,                    #Learning rate for perceptual model
           iterations=1000,         #Number of optimization steps for each batch
           randomize_noise = False,#Add noise to dlatents during optimization
           lazy_update = True):    #to not do more work than it needs
    encode_start = datetime.datetime.now()

    ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % src_dir)

    os.makedirs(generated_images_dir, exist_ok=True)
    os.makedirs(dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    batch_times = []
    batch_num = 0
    for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
        batch_num+=1
        start = datetime.datetime.now()
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        
        if lazy_update:
            if all(
                os.path.exists(os.path.join(generated_images_dir, f'{img_name}.png'))
                and os.path.exists(os.path.join(dlatent_dir, f'{img_name}.npy'))
                for img_name in names): 
                
                print(f"lazy update = {lazy_update} and all the images/.npy files for {names} already exist, so we are skipping this")
                continue

        perceptual_model.set_reference_images(images_batch)
#         op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr)
        for loss in tqdm(perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr),
                         total=iterations,
                         desc = ' '.join(names)
                        ):
            final_loss = loss
                         
#         pbar = tqdm(op, leave=False, total=iterations)
#         pbar.set_description(' '.join(names))
#         for loss in pbar:
# #             pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
#               final_loss = loss
        print(' '.join(names), ' loss:', final_loss)

        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()
        end = datetime.datetime.now()
        batch_times.append(end-start) #saving the time it took to process this batch
        print(f"Batch #{batch_num} took {end-start}")
    encode_end = datetime.datetime.now()
    print("Finished, yayyyy")
    print(f"total time = {encode_end-encode_start}")
    if len(batch_times) > 0:
        print(f"min batch time = {min(batch_times)}")
        print(f"max batch time = {max(batch_times)}")
    else:
        print("no batches were run this time!")
           
           
def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')

    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

        perceptual_model.set_reference_images(images_batch)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, learning_rate=args.lr)
        pbar = tqdm(op, leave=False, total=args.iterations)
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
        print(' '.join(names), ' loss:', loss)

        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()


if __name__ == "__main__":
    main()
